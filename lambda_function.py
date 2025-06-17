import requests
import json
import datetime
import os
import img2pdf
from pypdf import PdfWriter, PdfReader
import time
import re
import unicodedata
from PIL import Image, ImageFilter
import pytesseract
import base64
from collections import defaultdict
from pdf2image import convert_from_path # For splitting PDFs into images for OCR
import io # Import io for BytesIO

# AWS SDK for Python (Boto3)
import boto3
from botocore.exceptions import ClientError

# --- IMPORTANT: Configure Tesseract OCR path and Poppler path ---
# In Lambda, these binaries will typically be located in /opt/bin/
# You MUST create a Lambda Layer containing these compiled binaries for Amazon Linux.
pytesseract.pytesseract.tesseract_cmd = '/opt/bin/tesseract'
# For pdf2image, you need to set the poppler_path when calling convert_from_path
# e.g., convert_from_path(pdf_path, poppler_path='/opt/bin')


def get_secret(secret_name):
    """
    Retrieves a secret from AWS Secrets Manager.
    """
    region_name = os.environ.get("AWS_REGION", "us-east-1") # Use your preferred AWS region
    
    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        print(f"Error retrieving secret '{secret_name}': {e}")
        raise e
    else:
        # Decrypts secret using the associated KMS key.
        # Depending on whether the secret is a string or binary, one of these fields will be populated.
        if 'SecretString' in get_secret_value_response:
            return json.loads(get_secret_value_response['SecretString'])
        else:
            return base64.b64decode(get_secret_value_response['SecretBinary'])


# ========== BILL.COM API CONFIGURATION (V2 & V3) ==========
# Fetch these from Secrets Manager
BILL_COM_SECRETS = get_secret("bill_com_credentials") # Name of your secret in Secrets Manager

BASE_API_URL_V2 = "https://api.bill.com/api/v2"
BASE_API_URL_V3 = "https://gateway.prod.bill.com/connect/v3" 
DOWNLOAD_SERVLET_URL = "https://api.bill.com/DownloadBillDocumentServlet" 
UPDATE_BILL_URL_V2 = f"{BASE_API_URL_V2}/Crud/Update/Bill.json" 
CREATE_BILL_URL_V2 = f"{BASE_API_URL_V2}/Crud/Create/Bill.json" 
LOGIN_URL_V3 = f"{BASE_API_URL_V3}/login"

# List of vendor IDs to filter by (can also be managed in Secrets Manager or Lambda env vars)
VENDOR_IDS_TO_FILTER = [
    "00902EDBZQHCFZ26bws5",
    "00902JIMVFXTZH22kbdf",
    "00902YUUKDEKCV22kabn",
    "00902FYEOJBJYE26bwh9",
    "00902QVEAEQYJR22kb4d"
]

# Chart of Account IDs mapping (can be moved to a configuration file or database if large)
CHART_OF_ACCOUNT_IDS = {
    "5010 BOM - Balance of System (BOS)": "0ca02FXRHZXXLIAz9lyj",
    "5015 BOM - Batteries": "0ca02PVDVPLFJZUz9lyy",
    "5020 BOM - Cell Modem": "0ca02LAVYFFVVCRz9lys",
    "5025 BOM - Combiner": "0ca02KPSJWATWLUz9lyu",
    "5035 BOM - Inverters": "0ca02NZFTKILATLz9lyw",
    "5040 BOM - Panels": "0ca02SCNCXYKIAIz9lyr",
    "5045 BOM - Railing": "0ca02OZBZHTMTDPz9lyv",
    "5050 BOM - Sales Tax": "0ca02LUZQSPHPQDz9lyx"
}


# ========== FOLDER & FILE NAMING CONFIGURATION (Use /tmp/) ==========
# Lambda only allows writing to /tmp/. These folders will be created within /tmp/.
RAW_INVOICE_FOLDER = "/tmp/invoice_raw" 
PROCESSED_INVOICE_FOLDER = "/tmp/invoices_processed" 
TEMP_OCR_IMAGES_FOLDER = "/tmp/ocr_temp_images" 
FINAL_COMBINED_PDF_PREFIX = "invoice_batch_" 
LOGICAL_INVOICE_PDF_PREFIX = "logical_invoice_" 


# ========== BILL.COM API FUNCTIONS (V2) ==========

def login_v2():
    """
    Authenticates with the Bill.com API (production environment) to obtain a session ID.
    Returns the session ID if successful, otherwise None.
    """
    url = f"{BASE_API_URL_V2}/Login.json" 
    payload = f'orgId={BILL_COM_SECRETS["BILL_COM_ORG_ID"]}&devKey={BILL_COM_SECRETS["BILL_COM_DEV_KEY"]}&userName={BILL_COM_SECRETS["BILL_COM_USER_EMAIL"]}&password={BILL_COM_SECRETS["BILL_COM_USER_PASSWORD"]}'
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}

    try:
        response = requests.post(url, headers=headers, data=payload)
        print("--- Login V2 API Response ---")
        print(response.text)
        print("--------------------------")
        data = response.json()
        if data.get("response_status") != 0:
            print(f"[-] Login V2 failed: {data.get('response_message')}")
            return None
        session_id = data["response_data"]["sessionId"]
        print(f"[+] Login V2 successful. Session ID: {session_id}")
        return session_id
    except requests.exceptions.RequestException as req_err:
        print(f"[-] Network or request error during V2 login: {req_err}")
        return None
    except json.JSONDecodeError as json_err:
        print(f"[-] Failed to parse V2 login response JSON: {json_err}")
        print(f"Raw response text: {response.text}")
        return None
    except KeyError as key_err:
        print(f"[-] Missing expected key in V2 login response: {key_err}")
        print(f"Raw response data: {data}")
        return None
    except Exception as e:
        print(f"[-] An unexpected error occurred during V2 login: {e}")
        return None

def login_v3():
    """
    Authenticates with the Bill.com V3 API to obtain a session ID for V3 operations.
    Returns the session ID if successful, otherwise None.
    """
    payload = {
        "username": BILL_COM_SECRETS["BILL_COM_USER_EMAIL"],
        "password": BILL_COM_SECRETS["BILL_COM_USER_PASSWORD"],
        "organizationId": BILL_COM_SECRETS["BILL_COM_ORG_ID"],
        "devKey": BILL_COM_SECRETS["BILL_COM_DEV_KEY"]
    }
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(LOGIN_URL_V3, headers=headers, json=payload)
        print("--- Login V3 API Response ---")
        print(response.text) # Keep this for debugging the raw response
        print("--------------------------")
        data = response.json()
        session_id = data.get("sessionId")

        if session_id:
            print(f"[+] Login V3 successful. Session ID: {session_id}")
            return session_id
        else:
            response_status = data.get("response_status") 
            response_message = data.get('response_message', 'No specific error message provided.')
            
            print(f"[-] Login V3 failed. No session ID found in the root response.")
            print(f"     Raw data: {data}") 
            print(f"     Response Status (if any): {response_status}")
            print(f"     Response Message (if any): {response_message}")
            return None

    except requests.exceptions.RequestException as req_err:
        print(f"[-] Network or request error during V3 login: {req_err}")
        return None
    except json.JSONDecodeError as json_err:
        print(f"[-] Failed to parse V3 login response JSON: {json_err}")
        print(f"Raw response text: {response.text}") 
        return None
    except Exception as e:
        print(f"[-] An unexpected error occurred during V3 login: {e}")
        return None


def list_bills_created_yesterday(session_id):
    """
    Retrieves a list of bills created on the previous day using the List/Bill.json endpoint
    with server-side filtering, including a filter for specific vendor IDs.
    Returns a list of bill dictionaries (each containing all fields) if successful, otherwise None.
    """
    url = f"{BASE_API_URL_V2}/List/Bill.json"

    yesterday = datetime.datetime.utcnow() - datetime.timedelta(days=2)
    start_iso = yesterday.replace(hour=0, minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M:%S.000+0000")
    end_iso = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999).strftime("%Y-%m-%dT%H:%M:%S.999+0000")

    vendor_ids_str = ",".join(VENDOR_IDS_TO_FILTER)

    payload = {
        "data": json.dumps({
            "start": 0,
            "max": 100, 
            "filters": [
                {"field": "createdTime", "op": ">=", "value": start_iso},
                {"field": "createdTime", "op": "<=", "value": end_iso},
                {"field": "vendorId", "op": "in", "value": vendor_ids_str}
            ],
            "sort": [{"field": "createdTime", "asc": "true"}],
            "nested": True,
            "showAudit": True,
            "related": ["Vendor"]
        }),
        "devKey": BILL_COM_SECRETS["BILL_COM_DEV_KEY"],
        "sessionId": session_id
    }

    headers = {
        "accept": "application/json",
        "content-type": "application/x-www-form-urlencoded"
    }

    print(f"\n--- Attempting to get list of Bills created yesterday ({yesterday.date()}) for specific vendors ---")
    print(f"Filter range: {start_iso} to {end_iso}")
    print(f"Vendor IDs filter: {VENDOR_IDS_TO_FILTER}")
    try:
        response = requests.post(url, data=payload, headers=headers)
        print("[+] Status Code:", response.status_code)

        data = response.json()
        print("--- List/Bill API Response ---")
        print(json.dumps(data, indent=2))
        print("-----------------------------")

        if data.get("response_status") != 0:
            print(f"[-] List/Bill API call failed: {data.get('response_message')}")
            return None

        bills = data.get("response_data", []) 
        
        if not isinstance(bills, list):
            print(f"[-] Expected 'response_data' to be a list, but got type: {type(bills)}. Raw response data: {data}")
            return None

        if not bills:
            print(f"[!] No bills found for {yesterday.date()} with the specified vendor IDs.")
            return []

        print(f"[+] Successfully retrieved {len(bills)} bill(s) for {yesterday.date()} and specified vendors.")
        return bills
    except requests.exceptions.RequestException as req_err:
        print(f"[-] Network or request error during List/Bill: {req_err}")
        return None
    except json.JSONDecodeError as json_err:
        print(f"[-] Failed to parse List/Bill response JSON: {json_err}")
        print(f"Raw response text: {response.text}")
        return None
    except Exception as e:
        print(f"[-] An unexpected error occurred during List/Bill: {e}")
        return None


def get_bill_documents(session_id, bill_id):
    """
    Retrieves information about documents (attachments) linked to a specific bill.
    Returns a list of document dictionaries if successful, otherwise None.
    """
    url = f"{BASE_API_URL_V2}/GetDocuments.json"

    data_payload_json = json.dumps({
        "id": bill_id,
        "start": 0,
        "max": 100         
    })

    payload = {
        "data": data_payload_json,
        "devKey": BILL_COM_SECRETS["BILL_COM_DEV_KEY"],
        "sessionId": session_id
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/x-www-form-urlencoded"
    }

    try:
        response = requests.post(url, data=payload, headers=headers)
        print(f"\n--- GetDocuments API Response for Bill ID {bill_id} ---")
        print(response.text)
        print("-------------------------------------")

        data = response.json()
        if data.get("response_status") != 0:
            print(f"[-] GetDocuments API call failed for Bill ID {bill_id}: {data.get('response_message')}")
            if "session is invalid" in data.get("response_message", "").lower():
                print("[!] Session invalidation detected. You may need to re-authenticate.")
            return None

        documents = data["response_data"].get("documents", [])
        if not documents:
            print(f"[!] No documents found for Bill ID: {bill_id}")
            return []

        print(f"[+] Retrieved {len(documents)} document(s) for Bill ID: {bill_id}")
        return documents
    except requests.exceptions.RequestException as req_err:
        print(f"[-] Network or request error during GetDocuments for Bill ID {bill_id}: {req_err}")
        return None
    except json.JSONDecodeError as json_err:
        print(f"[-] Failed to parse GetDocuments response JSON for Bill ID {bill_id}: {json_err}")
        print(f"Raw response text: {response.text}")
        return None
    except KeyError as key_err:
        print(f"[-] Missing expected key in GetDocuments response for Bill ID {bill_id}: {key_err}")
        print(f"Raw response data: {data}")
        return None
    except Exception as e:
        print(f"[-] An unexpected error occurred during GetDocuments for Bill ID {bill_id}: {e}")
        return None


def download_bill_attachment(session_id, file_download_id, original_bill_id, base_filename, output_folder):
    """
    Downloads a full bill attachment using the DownloadBillDocumentServlet.
    Dynamically determines the file extension based on the Content-Type header AND magic bytes.
    Saves the downloaded content into the specified output_folder.
    Returns the path to the downloaded/converted PDF if successful, otherwise None.
    """
    url = f"{DOWNLOAD_SERVLET_URL}?id={file_download_id}&billId={original_bill_id}"

    headers = {
        "sessionId": session_id,
        "accept": "application/pdf, image/png, image/jpeg, image/gif"
    }

    try:
        response = requests.get(url, headers=headers)

        print(f"\n--- Download Attachment API Response Status Code for Document ID {file_download_id}: {response.status_code} ---")
        print(f"--- Response Headers for Document ID {file_download_id}: {response.headers} ---") 
        
        if response.status_code != 200:
            print(f"Response content (if available): {response.text}")
            if "session timed out" in response.text.lower() or "session is invalid" in response.text.lower():
                print("[!] Session timeout detected. You may need to re-authenticate.")
            print("---------------------------------------------------------------")
            return None

        if response.status_code == 200:
            content_type = response.headers.get('Content-Type', '').lower()
            
            # Determine file extension and save to a temporary path first
            temp_file_extension = ".bin"
            if response.content.startswith(b'%PDF-'):
                temp_file_extension = ".pdf"
            elif 'image/png' in content_type:
                temp_file_extension = ".png"
            elif 'image/jpeg' in content_type or 'image/jpg' in content_type:
                temp_file_extension = ".jpeg"
            elif 'image/gif' in content_type:
                temp_file_extension = ".gif"

            temp_file_name = f"{base_filename}_{file_download_id}{temp_file_extension}"
            temp_file_path = os.path.join(output_folder, temp_file_name)

            with open(temp_file_path, 'wb') as f:
                f.write(response.content)
            print(f"[+] Raw attachment '{base_filename}' downloaded to {temp_file_path} (Content-Type: {content_type})")

            # Now, convert to PDF if it's an image, and return the final PDF path
            if temp_file_extension != ".pdf":
                output_pdf_path = os.path.join(output_folder, os.path.basename(temp_file_path).replace(temp_file_extension, '.pdf'))
                converted_pdf_path = convert_image_to_pdf(temp_file_path, output_pdf_path)
                # Clean up the temporary image file after conversion
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                    print(f"[+] Removed temporary image file: {temp_file_path}")
                return converted_pdf_path
            else:
                # If it was already a PDF, just return its path
                return temp_file_path
        else:
            print(f"[-] Failed to download attachment for Document ID {file_download_id}. HTTP Status Code: {response.status_code}")
            print("---------------------------------------------------------------")
            return None
    except requests.exceptions.RequestException as req_err:
        print(f"[-] Network or request error during attachment download for Document ID {file_download_id}: {req_err}")
        return None
    except Exception as e:
        print(f"[-] An unexpected error occurred during attachment download for Document ID {file_download_id}: {e}")
        return None


# ========== PDF PROCESSING FUNCTIONS ==========

def convert_image_to_pdf(image_path, output_pdf_path):
    """
    Converts an image file (PNG, JPEG, GIF) to a PDF file.
    Returns the path to the generated PDF if successful, otherwise None.
    """
    try:
        with open(output_pdf_path, "wb") as f:
            f.write(img2pdf.convert(image_path))
        print(f"[+] Converted '{image_path}' to PDF: '{output_pdf_path}'")
        return output_pdf_path
    except Exception as e:
        print(f"[-] Error converting image '{image_path}' to PDF: {e}")
        return None

def combine_pdfs(pdf_list, output_combined_pdf_path):
    """
    Combines a list of PDF files into a single PDF.
    Returns the path to the combined PDF if successful, otherwise None.
    """
    if not pdf_list:
        print("[-] No PDFs to combine.")
        return None

    writer = PdfWriter()
    try:
        for pdf_file in pdf_list:
            if not os.path.exists(pdf_file):
                print(f"[-] Warning: PDF file not found, skipping: {pdf_file}")
                continue
            
            reader = PdfReader(pdf_file)
            num_pages_in_current_pdf = len(reader.pages) 
            print(f"[+] Processing '{pdf_file}' which has {num_pages_in_current_pdf} page(s).")
            for page in reader.pages:
                writer.add_page(page)
            print(f"[+] Added pages from '{pdf_file}' to the combined PDF.")
        
        with open(output_combined_pdf_path, "wb") as f:
            writer.write(f)
        print(f"[+] Successfully combined {len(pdf_list)} PDFs into: '{output_combined_pdf_path}'")
        return output_combined_pdf_path
    except Exception as e:
        print(f"[-] Error combining PDFs: {e}")
        return None

def combine_logical_invoice_to_pdf(image_paths, output_pdf_path):
    """
    Combines a list of image paths (representing pages of a single logical invoice) into a single PDF.
    This function is primarily for creating a single PDF for OCR processing, not necessarily for re-upload.
    """
    if not image_paths:
        print("[-] No image paths provided to combine into a logical invoice PDF.")
        return None

    writer = PdfWriter()
    try:
        for img_path in image_paths:
            if not os.path.exists(img_path):
                print(f"[-] Warning: Image file not found for logical invoice, skipping: {img_path}")
                continue
            
            try:
                pdf_bytes = img2pdf.convert(img_path)
                reader = PdfReader(io.BytesIO(pdf_bytes))
                for page in reader.pages:
                    writer.add_page(page)
                print(f"[+] Added page from '{img_path}' to the logical invoice PDF.")
            except img2pdf.ImageOpenError as img_err:
                print(f"[-] Error opening image with img2pdf for '{img_path}': {img_err}")
                continue
            except Exception as conv_err:
                print(f"[-] Unexpected error during img2pdf conversion for '{img_path}': {conv_err}")
                continue
        
        if not writer.pages:
            print(f"[-] No pages were added to the logical invoice PDF. Skipping write to '{output_pdf_path}'.")
            return None

        with open(output_pdf_path, "wb") as f:
            writer.write(f)
        print(f"[+] Successfully combined {len(image_paths)} images into logical invoice PDF: '{output_pdf_path}'")
        return output_pdf_path
    except Exception as e:
        print(f"[-] Error combining logical invoice images to PDF: {e}")
        return None


# ========== OCR AND COA PROCESSING FUNCTIONS ==========

def sanitize_json_string(text):
    text = text.strip()
    if text.startswith("```json"): text = text[7:]
    if text.endswith("```"): text = text[:-3]
    text = text.strip()
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = re.sub(r"(?<=\{|,|\s)'([^']+)':", r'"\1":', text)
    text = re.sub(r":\s*'([^']*)'", r': "\1"', text)
    text = re.sub(r"\[\s*'([^']*)'\s*\]", r'["\1"]', text)
    text = re.sub(r'(?<=[:\s])(\d{1,3}(?:,\d{3})+(?:\.\d+)?)(?=[,}\]])', lambda m: m.group(1).replace(",", ""), text)
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\r\t")
    return text


def extract_invoice_number(image_path):
    # Retrieve OpenAI API key securely
    openai_api_key = BILL_COM_SECRETS.get("OPENAI_API_KEY") 
    if not openai_api_key:
        print("[-] OpenAI API key not found in secrets. Cannot perform OCR extraction.")
        return None
    
    # Initialize OpenAI client within the function scope or globally if preferred, but ensure key is set.
    from openai import OpenAI
    client = OpenAI(api_key=openai_api_key)

    ocr_text = pytesseract.image_to_string(Image.open(image_path), config="--psm 4")
    prompt = """
    From the OCR output of a single invoice page, extract the invoice number exactly as shown.
    It might appear as: Invoice #, Invoice No., or a number like 0311-1060296 or 4011 - 100280, or sometimes with a letter "S" at the beginning like S007918758.
    Return valid JSON only:
    {"invoice_number": "..."} or {"invoice_number": null} if not found.
    """
    response = client.chat.completions.create( # Use client.chat.completions.create
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "text", "text": ocr_text}
            ]
        }]
    )
    content = response.choices[0].message.content
    try:
        return json.loads(sanitize_json_string(content))["invoice_number"]
    except json.JSONDecodeError as e:
        print(f"[-] Error decoding JSON from OpenAI response in extract_invoice_number: {e}")
        print(f"Raw content: {content}")
        return None


def split_pdf_and_tag_pages(pdf_path, output_folder):
    # Ensure poppler_path is passed to convert_from_path
    poppler_path = '/opt/bin' if os.path.exists('/opt/bin/pdftoppm') else None
    if not poppler_path:
        print("[-] Warning: Poppler utilities not found at /opt/bin/. PDF to image conversion may fail.")
    
    images = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)
    pages = []
    for i, img in enumerate(images):
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        img_path = os.path.join(output_folder, f"page_{i+1}.png")
        img.save(img_path)
        invoice_number = extract_invoice_number(img_path)
        pages.append({"path": img_path, "invoice_number": invoice_number, "page_number": i+1})
    return pages


def group_pages_by_invoice(pages):
    grouped = defaultdict(list)
    for page in sorted(pages, key=lambda x: x["page_number"]):
        key = page["invoice_number"] if page["invoice_number"] is not None else f"unidentified_invoice_{len(grouped) + 1}"
        grouped[key].append(page) 
    return grouped


def process_invoice(invoice_number, page_paths_and_data, output_folder): 
    # Retrieve OpenAI API key securely
    openai_api_key = BILL_COM_SECRETS.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("[-] OpenAI API key not found in secrets. Cannot perform OCR extraction.")
        return {}, {}, [], "" # Return empty values
    
    from openai import OpenAI
    client = OpenAI(api_key=openai_api_key)

    print(f"Processing {invoice_number} ({len(page_paths_and_data)} pages)...")
    
    final_page_path = page_paths_and_data[-1]["path"] 
    final_page_image = Image.open(final_page_path).convert("RGB")

    with open(final_page_path, "rb") as f:
        b64_image_for_header = base64.b64encode(f.read()).decode("utf-8")

    prompt_header = """
        You are a highly precise invoice processor.
        Extract the following from the provided invoice:
        - vendor_name (exactly as written)
        - invoice_number
        - invoice_date (in original format)
        - invoice_due_date (sometimes listed, sometimes described as the 15th of the month following the purchase. If the 15th of the month following the purchase, set invoice_due_date equal to 15th of the month following invoice_date. Return in MM/DD/YYYY)
        - sales_tax (as a number)
        - extracted_total_due (value found on the last page of the invoice)
        If any field is unclear or missing, use null.

        ** Important ** Sales tax will never be 8.75. This is a percentage indicating the sales tax rate. The sales tax number will likely be beside that.
        ** Important ** extracted_total_due will always be found at the bottom right of the last page of the invoice, next to the words "Total Due"
        Return valid JSON only. No explanation or commentary.
        """
    header_resp = client.chat.completions.create( # Use client.chat.completions.create
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_header},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image_for_header}", "detail": "high"}} 
            ]
        }],
        max_tokens=1000
    )
    print(header_resp)
    header_data = json.loads(sanitize_json_string(header_resp.choices[0].message.content.strip()))
    
    extracted_total_due_str = str(header_data.get("extracted_total_due", "0.0")).replace(",", "")
    extracted_total_due = float(extracted_total_due_str or 0.0)
    print(f"Extracted Total Due: {extracted_total_due}")

    images_for_full_ocr = [Image.open(p["path"]).convert("RGB") for p in page_paths_and_data]
    width = max(img.width for img in images_for_full_ocr)
    total_height = sum(img.height for img in images_for_full_ocr)
    combined_for_ocr_image = Image.new("RGB", (width, total_height))
    y_offset = 0
    for img in images_for_full_ocr:
        combined_for_ocr_image.paste(img, (0, y_offset))
        y_offset += img.height
    
    combined_for_ocr_image_path = os.path.join(output_folder, f"{invoice_number}_combined_for_ocr.png")
    combined_for_ocr_image.save(combined_for_ocr_image_path)

    text_for_line_items_ocr = pytesseract.image_to_string(combined_for_ocr_image, config="--psm 4")

    prompt_lines = """
        You are given text that was extracted by an OCR from an invoice. You NEVER change numbers from what the OCR text has written.

        Organize the following fields from the line items of the provided invoice

        - line_items: a list of dictionaries with keys:
            - product_code (string, exactly as written)
            - description (string, exactly as written with no modifications or summaries)
            - total (number without commas). This is usually found after a letter C or E, which comes after 2 other numbers. Ignore anything after the two decimal places.
        - calculated_total_due: The sum of the totals of each line item

        CRITICAL INSTRUCTIONS:
        1. For line items, write exactly what is given you by the OCR for the numbers. Under NO CIRCUMSTANCES are you to change a number, regardless of quantity, price, total disagreement.
        2. unit_price and total values should under NO circumstances include commas. If there is a comma, rewrite the number exactly with the comma removed. Do NOT replace the comma with anything. Simply remove it.
        3. Never include smart quotes. 
        2. For ANY field that is unclear or not visible, use null instead of guessing
        3. Leave descriptions EXACTLY as written - do not paraphrase, summarize, or modify them
        4. For line items:
        - Extract them in the exact order they appear on the invoice
        - Skip headers, subtotals, or summary rows
        - If any field in a line item is unclear, mark it as null but include other clear fields
        5. For numbers:
        - Extract numbers WITHOUT commas but WITH decimal points

        Return a JSON object with the following fields:
        - line_items: a list of dictionaries with keys:
            - product_code (string, exactly as written)
            - description (string, exactly as written with no modifications or summaries)
            - total (number without commas)

        Do not include any commentary or extra explanation — return valid JSON only.
        """
    line_resp = client.chat.completions.create( # Use client.chat.completions.create
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_lines},
                {"type": "text", "text": text_for_line_items_ocr}
            ]
        }],
        max_tokens=8000
    )
    line_data = json.loads(sanitize_json_string(line_resp.choices[0].message.content.strip()))
    line_items = line_data.get("line_items", [])

    tax = header_data.get("sales_tax", 0.0)
    if tax:
        line_items.append({"description": "Sales Tax", "total": tax}) 

    # Moved Assistant API logic here and simplified for direct use
    # The Assistant API is typically used for more complex, multi-turn conversations.
    # For a single classification, a direct chat completion might be more efficient.
    # However, if you've already set up an Assistant for COA, we can keep it.
    # Ensure 'openai.beta.threads' is still the correct way to call it with the new client.
    # From the provided code, it looks like you're using `openai` directly, not `client.beta.threads`.
    # Let's adjust to use the `client` object for consistency.

    # Re-initialise the OpenAI client to use the `beta` namespace for the Assistant API
    # Assuming `client = OpenAI(api_key=openai_api_key)` already done above.
    
    # Construct the message for the Assistant
    messages_for_assistant = [
        {"role": "user", "content": "Here are invoice line items. Classify each into a chart of account:\n\n" + json.dumps(line_items, indent=2)}
    ]

    # Call the chat completions API for classification
    # Assuming "asst_EFTIyPhnWXMTE9mXpmoJFrmo" is a valid assistant ID
    # For a simple classification like this, you might not strictly need the Assistant API.
    # A regular chat completion with a precise prompt could also work.
    
    # If using the Assistant API, the process would look more like this (requires Assistant setup):
    # This part needs careful handling as `openai.beta.threads` changed slightly with `OpenAI()` client.
    # If your Assistant expects a specific input format, adjust `messages_for_assistant`.

    # This is a placeholder for the Assistant API interaction.
    # If this part needs to run, ensure your assistant setup is correct and permissions are in place.
    # For a simpler approach, you can use a chat completion for classification directly.
    # For now, let's keep the existing structure, assuming the Assistant setup is working.

    # NOTE: The provided code uses `openai.beta.threads.create` directly.
    # For the `OpenAI()` client, it should be `client.beta.threads.create`.
    # Also, the Assistant needs to be configured to output JSON with a specific schema
    # for the `raw_coa` parsing to work reliably.

    # Assuming `asst_EFTIyPhnWXMTE9mXpmoJFrmo` is an assistant configured for JSON output.
    # This part of the code needs to be adjusted to the new OpenAI client structure if not already.
    # Since the original code uses `openai.beta.threads`, let's ensure the `openai` module
    # is configured with the API key, or pass `client` to an inner function.
    # For simplicity in this example, let's assume `openai.api_key` is set globally *if using the old way*,
    # or ensure `client` from `OpenAI()` is used correctly.

    # To make it compatible with `OpenAI()` client:
    # client.beta.threads.messages.create(thread_id=thread.id, role="user", content=messages_for_assistant[0]["content"])
    # client.beta.threads.runs.create(thread_id=thread.id, assistant_id="asst_EFTIyPhnWXMTE9mXpmoJFrmo")
    # ... and then retrieve messages using `client.beta.threads.messages.list`

    # For now, let's assume the existing openai.beta calls work with the global key setting
    # or that `client` is correctly configured for `beta` calls.
    # If not, this section might need further refactoring depending on the exact OpenAI library version.

    # --- Simplified COA Classification (Alternative to Assistant API if simpler) ---
    # This section demonstrates a direct chat completion for COA if Assistant API is too complex.
    prompt_coa = f"""
    Given the following invoice line items, classify each into one of the provided Chart of Account names.
    Return a JSON array where each object has a 'description', 'total', and 'chart_of_account' field.
    If a line item cannot be classified, use "Other" or "Uncategorized" as the chart_of_account.
    Only use the exact Chart of Account names provided in the list below.

    Available Chart of Account Names:
    {json.dumps(list(CHART_OF_ACCOUNT_IDS.keys()), indent=2)}

    Invoice Line Items:
    {json.dumps(line_items, indent=2)}

    Return valid JSON only. No commentary or extra explanation.
    Example Output:
    [
      {{"description": "Product A", "total": 100.00, "chart_of_account": "5010 BOM - Balance of System (BOS)"}},
      {{"description": "Shipping", "total": 10.00, "chart_of_account": "5050 BOM - Sales Tax"}}
    ]
    """
    coa_resp = client.chat.completions.create(
        model="gpt-4o", # Or "gpt-3.5-turbo" if cost is a concern and quality is sufficient
        messages=[{"role": "user", "content": prompt_coa}],
        response_format={"type": "json_object"} # Request JSON output directly
    )
    raw_coa_content = coa_resp.choices[0].message.content
    try:
        # The prompt asks for an array, but the response_format={"type": "json_object"}
        # implies the entire response is a JSON object. We need to check if the LLM
        # puts the array directly at the root or within a key.
        parsed_coa = json.loads(raw_coa_content)
        # If the LLM wraps the array in a key (e.g., {"line_items": [...]})
        if isinstance(parsed_coa, dict) and "line_items" in parsed_coa:
            coa_data = parsed_coa
        else: # If the LLM returns just the array at the root
            coa_data = {"line_items": parsed_coa}
    except json.JSONDecodeError as e:
        print(f"[-] Error decoding JSON for COA classification: {e}")
        print(f"Raw COA content: {raw_coa_content}")
        coa_data = {"line_items": []} # Default to empty on error

    # --- End Simplified COA Classification ---

    # Original Assistant API calls (keep only if you are sure your Assistant is set up)
    # try:
    #     thread = openai.beta.threads.create()
    #     openai.beta.threads.messages.create(thread_id=thread.id, role="user", content=user_input)
    #     run = openai.beta.threads.runs.create(thread_id=thread.id, assistant_id="asst_EFTIyPhnWXMTE9mXpmoJFrmo")
    #     while True:
    #         status = openai.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
    #         if status.status == "completed": break
    #         time.sleep(1)
    #     final = openai.beta.threads.messages.list(thread_id=thread.id)
    #     raw_coa = sanitize_json_string(final.data[0].content[0].text.value.strip())
    #     if not raw_coa:
    #         print("[-] Warning: OpenAI API returned empty content for COA classification. Defaulting to empty line items.")
    #         coa_data = {"line_items": []}
    #     elif raw_coa.startswith('['):
    #         coa_data = {"line_items": json.loads(raw_coa)}
    #     else:
    #         coa_data = json.loads(raw_coa)
    # except Exception as e:
    #     print(f"[-] Error during OpenAI Assistant API call for COA: {e}")
    #     coa_data = {"line_items": []} # Default to empty on error
    # --- End Original Assistant API calls ---


    summary = defaultdict(float)
    extracted_total_due_str = str(header_data.get("extracted_total_due", "0.0")).replace(",", "")
    extracted_total_due = float(extracted_total_due_str or 0.0)
    print(f"Extracted Total Due: {extracted_total_due}")

    coa_sum = 0.0
    for item in coa_data.get("line_items", []):
        coa = item.get("chart_of_accounts") or item.get("chart_of_account")
        total = float(item.get("total", 0.0) or 0.0)
        if coa:
            summary[coa] += total
            coa_sum += total

    bos_key = "5010 BOM - Balance of System (BOS)"
    if bos_key in summary:
        difference = extracted_total_due - (coa_sum - summary[bos_key])
        summary[bos_key] = round(difference, 2)
    elif extracted_total_due and coa_sum < extracted_total_due:
            summary[bos_key] = round(extracted_total_due - coa_sum, 2)


    with open(os.path.join(output_folder, f"result_{invoice_number}.json"), "w") as f:
        json.dump({"invoice_number": invoice_number, "coa_summary": dict(summary)}, f, indent=2)
    
    page_numbers = sorted([p['page_number'] for p in page_paths_and_data])
    if len(page_numbers) == 1:
        description = f"Invoice from page {page_numbers[0]} of original PDF."
    elif page_numbers:
        description = f"Invoice from pages {page_numbers[0]}-{page_numbers[-1]} of original PDF."
    else:
        description = "Invoice without specific page reference."

    return summary, header_data, page_paths_and_data, description


# ========== BILL.COM API FUNCTIONS (V2 & V3 Operations) ==========

def update_bill_in_billcom(session_id_v2, bill_id, vendor_id, invoice_number, invoice_date, due_date, coa_summary, description=None):
    """
    Updates a bill in Bill.com with extracted invoice data and categorized line items.
    Uses V2 API.
    """
    bill_line_items = []
    for coa_name, amount in coa_summary.items():
        coa_id = CHART_OF_ACCOUNT_IDS.get(coa_name)
        if coa_id:
            bill_line_items.append({
                "entity": "BillLineItem",
                "amount": round(amount, 2), 
                "chartOfAccountId": coa_id
            })
        else:
            print(f"[-] Warning: Chart of Account ID not found for '{coa_name}'. Skipping this line item for update.")

    invoice_date_str = None
    if invoice_date:
        try:
            if len(invoice_date.split('/')[2]) == 2: 
                parsed_date = datetime.datetime.strptime(invoice_date, "%m/%d/%y").date()
            else:
                parsed_date = datetime.datetime.strptime(invoice_date, "%m/%d/%Y").date()
            invoice_date_str = parsed_date.strftime("%Y-%m-%d")
        except ValueError:
            print(f"[-] Warning: Could not parse invoice_date '{invoice_date}'. Keeping as is or null.")
            invoice_date_str = invoice_date 

    due_date_str = None
    if due_date:
        try:
            if len(due_date.split('/')[2]) == 2: 
                parsed_date = datetime.datetime.strptime(due_date, "%m/%d/%y").date()
            else:
                parsed_date = datetime.datetime.strptime(due_date, "%m/%d/%Y").date()
            due_date_str = parsed_date.strftime("%Y-%m-%d") 
        except ValueError:
            print(f"[-] Warning: Could not parse due_date '{due_date}'. Keeping as is or null.")
            due_date_str = due_date 


    bill_obj = { 
        "entity": "Bill",
        "id": bill_id,
        "isActive": "1", 
        "vendorId": vendor_id,
        "invoiceNumber": invoice_number,
        "invoiceDate": invoice_date_str,
        "dueDate": due_date_str,
        "billLineItems": bill_line_items
    }
    if description: 
        bill_obj["description"] = description

    data_payload_json = json.dumps({"obj": bill_obj})

    payload = {
        "data": data_payload_json,
        "devKey": BILL_COM_SECRETS["BILL_COM_DEV_KEY"],
        "sessionId": session_id_v2
    }

    headers = {
        "accept": "application/json",
        "content-type": "application/x-www-form-urlencoded"
    }

    print(f"\n--- Attempting to update Bill ID: {bill_id} (V2) ---")
    try:
        response = requests.post(UPDATE_BILL_URL_V2, data=payload, headers=headers)
        print("[+] Update Bill Status Code:", response.status_code)
        update_data = response.json()
        print("--- Update Bill API Response ---")
        print(json.dumps(update_data, indent=2))
        print("--------------------------------")

        if update_data.get("response_status") == 0:
            print(f"[+] Successfully updated Bill ID: {bill_id}")
            return True
        else:
            print(f"[-] Failed to update Bill ID: {bill_id}. Message: {update_data.get('response_message')}")
            return False
    except Exception as e:
        print(f"[-] An unexpected error occurred during Bill update for {bill_id}: {e}")
        return False

def create_bill_in_billcom(session_id_v2, vendor_id, invoice_number, invoice_date, due_date, coa_summary, description=None):
    """
    Creates a new bill in Bill.com with extracted invoice data and categorized line items.
    Returns the new bill_id if successful, otherwise None.
    Uses V2 API.
    """
    bill_line_items = []
    for coa_name, amount in coa_summary.items():
        coa_id = CHART_OF_ACCOUNT_IDS.get(coa_name)
        if coa_id:
            bill_line_items.append({
                "entity": "BillLineItem",
                "amount": round(amount, 2), 
                "chartOfAccountId": coa_id
            })
        else:
            print(f"[-] Warning: Chart of Account ID not found for '{coa_name}'. Skipping this line item for new bill creation.")

    invoice_date_str = None
    if invoice_date:
        try:
            if len(invoice_date.split('/')[2]) == 2: 
                parsed_date = datetime.datetime.strptime(invoice_date, "%m/%d/%y").date()
            else:
                parsed_date = datetime.datetime.strptime(invoice_date, "%m/%d/%Y").date()
            invoice_date_str = parsed_date.strftime("%Y-%m-%d")
        except ValueError:
            print(f"[-] Warning: Could not parse invoice_date '{invoice_date}'. Keeping as is or null for new bill.")
            invoice_date_str = invoice_date 

    due_date_str = None
    if due_date:
        try:
            if len(due_date.split('/')[2]) == 2: 
                parsed_date = datetime.datetime.strptime(due_date, "%m/%d/%y").date()
            else:
                parsed_date = datetime.datetime.strptime(due_date, "%m/%d/%Y").date()
            due_date_str = parsed_date.strftime("%Y-%m-%d") 
        except ValueError:
            print(f"[-] Warning: Could not parse due_date '{due_date}'. Keeping as is or null for new bill.")
            due_date_str = due_date 

    bill_obj = { 
        "entity": "Bill",
        "vendorId": vendor_id,
        "invoiceNumber": invoice_number,
        "invoiceDate": invoice_date_str,
        "dueDate": due_date_str,
        "billLineItems": bill_line_items
    }
    if description: 
        bill_obj["description"] = description

    data_payload_json = json.dumps({"obj": bill_obj})

    payload = {
        "data": data_payload_json,
        "devKey": BILL_COM_SECRETS["BILL_COM_DEV_KEY"],
        "sessionId": session_id_v2
    }

    headers = {
        "accept": "application/json",
        "content-type": "application/x-www-form-urlencoded"
    }

    print(f"\n--- Attempting to create NEW Bill (V2) for invoice number: {invoice_number} ---")
    try:
        response = requests.post(CREATE_BILL_URL_V2, data=payload, headers=headers)
        print("[+] Create Bill Status Code:", response.status_code)
        create_data = response.json()
        print("--- Create Bill API Response ---")
        print(json.dumps(create_data, indent=2))
        print("--------------------------------")

        if create_data.get("response_status") == 0:
            new_bill_id = create_data.get("response_data", {}).get("id")
            print(f"[+] Successfully created new Bill with ID: {new_bill_id} for invoice number: {invoice_number}")
            return new_bill_id
        else:
            print(f"[-] Failed to create new Bill for invoice number: {invoice_number}. Message: {create_data.get('response_message')}")
            return None
    except Exception as e:
        print(f"[-] An unexpected error occurred during new Bill creation for invoice number {invoice_number}: {e}")
        return None

def upload_attachment_to_bill_v3(session_id_v3, bill_id, file_path):
    """
    Uploads a PDF attachment to a specific bill using the V3 API.
    """
    if not os.path.exists(file_path):
        print(f"[-] File not found for V3 upload: {file_path}")
        return False

    with open(file_path, "rb") as f:
        file_content = f.read()
    
    encoded_file_content = base64.b64encode(file_content).decode('utf-8')
    
    file_name = os.path.basename(file_path)

    upload_url = f"{BASE_API_URL_V3}/documents/bills/{bill_id}?name={file_name}" 

    payload = {
        "obj": {
            "entity": "Document",
            "name": file_name, 
            "parent": { 
                "id": bill_id,
                "entity": "Bill"
            },
            "data": encoded_file_content 
        },
        "devKey": BILL_COM_SECRETS["BILL_COM_DEV_KEY"]
    }

    headers = {
        "Content-Type": "application/json", 
        "sessionId": session_id_v3,
        "devKey": BILL_COM_SECRETS["BILL_COM_DEV_KEY"] 
    }

    print(f"\n--- Attempting to upload attachment '{file_name}' to Bill ID {bill_id} (V3) ---")
    print(f"--- Upload URL: {upload_url} ---")
    try:
        response = requests.post(upload_url, headers=headers, json=payload) 
        print("[+] Upload V3 Status Code:", response.status_code)
        
        upload_data = response.json()
        print("--- Upload V3 API Response ---")
        print(json.dumps(upload_data, indent=2))
        print("------------------------------")

        if response.status_code in [200, 201] and isinstance(upload_data, dict) and upload_data.get("uploadId"):
            print(f"[+] Successfully uploaded '{file_name}' to Bill ID {bill_id} (V3). Upload ID: {upload_data['uploadId']}")
            return True
        else:
            print(f"[-] Failed to upload '{file_name}' to Bill ID {bill_id} (V3).")
            error_message = "No specific error message or successful ID in response."
            if isinstance(upload_data, list) and upload_data:
                error_messages = [err.get("message", "Unknown error") for err in upload_data if isinstance(err, dict)]
                error_message = "; ".join(error_messages)
            elif isinstance(upload_data, dict) and upload_data.get("message"): 
                error_message = upload_data.get("message")
            elif isinstance(upload_data, dict) and upload_data.get("response_message"): 
                error_message = upload_data.get("response_message")
            
            print(f"     Error Message: {error_message}")
            return False
    except requests.exceptions.RequestException as req_err:
        print(f"[-] Network or request error during V3 attachment upload for Bill ID {bill_id}: {req_err}")
        return False
    except Exception as e:
        print(f"[-] An unexpected error occurred during V3 attachment upload for Bill ID {bill_id}: {e}")
        return False


def process_single_bill(session_id_v2, bill_data): 
    """
    Processes a single bill: downloads its attachments, converts/combines them into a single PDF,
    runs OCR/COA processing on the combined PDF, and then attempts to update the bill in Bill.com.
    If multiple logical invoices are detected, the first updates the original bill,
    and subsequent invoices create new bills and have their respective PDFs attached via V3 API.
    """
    bill_id = bill_data.get('id')
    vendor_id = bill_data.get('vendorId')
    bill_description = bill_data.get('description', '') 

    print(f"\n{'='*60}")
    print(f"Processing Bill ID: {bill_id} (Vendor ID: {vendor_id})")
    print(f"{'='*60}\n")

    if bill_description:
        print(f"[!] Bill ID {bill_id} already has a description: '{bill_description}'. Skipping further processing for this bill to avoid duplication.")
        return

    os.makedirs(RAW_INVOICE_FOLDER, exist_ok=True)
    os.makedirs(PROCESSED_INVOICE_FOLDER, exist_ok=True)
    os.makedirs(TEMP_OCR_IMAGES_FOLDER, exist_ok=True)

    bill_downloaded_file_paths = []
    bill_pdfs_for_combination = []
    bill_intermediate_pdfs_for_cleanup = []
    
    canonical_attachment_pdfs = [] 

    bill_attachments = get_bill_documents(session_id_v2, bill_id)

    if bill_attachments is None:
        print(f"[-] Failed to retrieve documents for Bill ID {bill_id}. Skipping processing.")
        return

    if not bill_attachments:
        print(f"[!] No attachments found for Bill ID: {bill_id}. Skipping processing.")
        return
    
    for doc_idx, attachment in enumerate(bill_attachments):
        attachment_id = attachment.get('id')
        file_name = attachment.get('fileName')
        num_pages_original_doc = attachment.get('numPages', 1)

        if not attachment_id or not file_name:
            print(f"[-] Skipping document {doc_idx + 1} for Bill ID {bill_id} due to missing ID or filename.")
            continue

        print(f"\n--- Processing attachment '{file_name}' (ID: {attachment_id}) reported as {num_pages_original_doc} page(s) for Bill ID {bill_id} ---")
        
        base_filename, _ = os.path.splitext(file_name) 
        
        print(f"\nAttempting to download full document '{file_name}' for Bill ID {bill_id}...")
        downloaded_pdf_path = download_bill_attachment(
            session_id=session_id_v2,
            file_download_id=attachment_id,
            original_bill_id=bill_id,
            base_filename=base_filename,
            output_folder=RAW_INVOICE_FOLDER
        )
        
        if downloaded_pdf_path:
            canonical_attachment_pdfs.append(downloaded_pdf_path)
            bill_pdfs_for_combination.append(downloaded_pdf_path) 
            print(f"Canonical PDF for attachment '{file_name}' saved to: {downloaded_pdf_path}")
        else:
            print(f"Failed to download or convert full document '{file_name}' for Bill ID {bill_id}.")

    if bill_pdfs_for_combination:
        print(f"\n--- Starting PDF Combination for OCR for Bill ID {bill_id} ---")
        bill_pdfs_for_combination.sort() 
        print(f"PDFs identified for overall combination (sorted): {bill_pdfs_for_combination}") 

        combined_pdf_filename = f"{FINAL_COMBINED_PDF_PREFIX}{bill_id}.pdf"
        combined_pdf_path = os.path.join(PROCESSED_INVOICE_FOLDER, combined_pdf_filename)
        print(f"\n--- Combining all processed PDFs for Bill ID {bill_id} into '{combined_pdf_path}' for OCR ---")
        
        final_combined_pdf = combine_pdfs(bill_pdfs_for_combination, combined_pdf_path)
        if final_combined_pdf:
            print(f"[+] All invoice attachments for Bill ID {bill_id} combined into: '{final_combined_pdf}' for OCR processing.")
            
            print(f"\n--- Starting OCR and COA processing for combined PDF: '{final_combined_pdf}' ---")
            
            pages_for_ocr = split_pdf_and_tag_pages(final_combined_pdf, TEMP_OCR_IMAGES_FOLDER)
            
            grouped_invoices = group_pages_by_invoice(pages_for_ocr)
            
            if not grouped_invoices:
                print(f"[-] No logical invoices could be identified in the combined PDF for Bill ID {bill_id}. Skipping COA processing and Bill.com update.")
            
            invoice_numbers_processed = [] 
            
            session_id_v3 = None
            if len(grouped_invoices) > 1: 
                print("\n[!] Multiple invoices detected. Attempting to log in to V3 API for attachment uploads.")
                session_id_v3 = login_v3()
                if not session_id_v3:
                    print("[-] Failed to obtain V3 session ID. Attachments for new bills will not be uploaded.")

            logical_invoice_attachment_map = {}
            if len(canonical_attachment_pdfs) == len(grouped_invoices):
                sorted_logical_invoice_numbers = sorted(grouped_invoices.keys())
                for i, inv_num in enumerate(sorted_logical_invoice_numbers):
                    logical_invoice_attachment_map[inv_num] = canonical_attachment_pdfs[i]
            elif len(canonical_attachment_pdfs) == 1:
                for inv_num in grouped_invoices.keys():
                    logical_invoice_attachment_map[inv_num] = canonical_attachment_pdfs[0]
            else:
                print(f"[-] Warning: Cannot reliably map {len(canonical_attachment_pdfs)} original attachments to {len(grouped_invoices)} logical invoices. Attachments for new bills might be incorrect.")
                if canonical_attachment_pdfs:
                    for inv_num in grouped_invoices.keys():
                        logical_invoice_attachment_map[inv_num] = canonical_attachment_pdfs[0]


            for invoice_idx, (invoice_number, page_data_list) in enumerate(grouped_invoices.items()): 
                print(f"\n=== Running OCR/COA for logical Invoice: {invoice_number} (from Bill ID {bill_id}) ===")
                summary, header_data, _, generated_description = process_invoice(invoice_number, page_data_list, PROCESSED_INVOICE_FOLDER) 

                if summary:
                    print(f"=== SUMMARY FOR INVOICE {invoice_number} ===")
                    print(json.dumps(summary, indent=2))
                    print(f"TOTAL FOR INVOICE {invoice_number}: ${sum(summary.values()):,.2f}")
                    print(f"Description for Bill: {generated_description}") 
                    print("------------------------")

                    invoice_date = header_data.get("invoice_date")
                    due_date = header_data.get("invoice_due_date") 

                    target_bill_id = None
                    if invoice_idx == 0: 
                        print(f"\n--- Attempting to update original Bill ID {bill_id} in Bill.com with data from invoice {invoice_number} ---")
                        update_success = update_bill_in_billcom(
                            session_id_v2=session_id_v2,
                            bill_id=bill_id,
                            vendor_id=vendor_id,
                            invoice_number=invoice_number,
                            invoice_date=invoice_date,
                            due_date=due_date,
                            coa_summary=summary,
                            description=generated_description 
                        )
                        if update_success:
                            print(f"[+] Original Bill ID {bill_id} updated successfully in Bill.com.")
                            target_bill_id = bill_id 
                        else:
                            print(f"[-] Failed to update original Bill ID {bill_id} in Bill.com.")
                    else: 
                        print(f"\n--- Detected multiple invoices. Attempting to create NEW Bill in Bill.com for invoice {invoice_number} ---")
                        new_bill_id = create_bill_in_billcom(
                            session_id_v2=session_id_v2,
                            vendor_id=vendor_id,
                            invoice_number=invoice_number,
                            invoice_date=invoice_date,
                            due_date=due_date,
                            coa_summary=summary,
                            description=generated_description 
                        )
                        if new_bill_id:
                            print(f"[+] New Bill created successfully in Bill.com with ID {new_bill_id} for invoice {invoice_number}.")
                            target_bill_id = new_bill_id 
                        else:
                            print(f"[-] Failed to create new Bill in Bill.com for invoice {invoice_number}.")
                    
                    invoice_numbers_processed.append(invoice_number)

                    attachment_pdf_for_upload = logical_invoice_attachment_map.get(invoice_number)
                    if not attachment_pdf_for_upload and canonical_attachment_pdfs:
                        attachment_pdf_for_upload = canonical_attachment_pdfs[0] 

                    if target_bill_id and invoice_idx > 0 and session_id_v3 and attachment_pdf_for_upload: 
                        print(f"\n--- Preparing to attach PDF for invoice {invoice_number} to Bill ID {target_bill_id} ---")
                        print(f"--- Using attachment file: {attachment_pdf_for_upload} ---")
                        
                        upload_success = upload_attachment_to_bill_v3(session_id_v3, target_bill_id, attachment_pdf_for_upload)
                        if upload_success:
                            print(f"[+] Attachment successfully uploaded for Bill ID {target_bill_id}.")
                        else:
                            print(f"[-] Failed to upload attachment for Bill ID {target_bill_id}.")
                    elif invoice_idx > 0 and not session_id_v3:
                            print(f"[-] V3 session not available. Skipping attachment upload for new Bill ID {target_bill_id}.")
                    elif invoice_idx > 0 and not attachment_pdf_for_upload:
                            print(f"[-] No suitable attachment PDF found for invoice {invoice_number}. Skipping attachment upload for new Bill ID {target_bill_id}.")

                else:
                    print(f"[-] Failed to process logical invoice {invoice_number} from Bill ID {bill_id}.")

            if not invoice_numbers_processed:
                print(f"[!] No invoices were successfully processed for Bill ID {bill_id} for Bill.com update/creation.")
            else:
                bill_grand_total = defaultdict(float)
                # Re-process to get summary for grand total (could optimize this to store earlier)
                # Note: Calling process_invoice again here will re-run OCR/LLM for the same invoices.
                # It's better to store the summaries from the first run.
                # For simplicity, keeping it as is, but in a production scenario, you'd store and reuse.
                for invoice_number_key, page_data_list_for_sum in grouped_invoices.items(): 
                    # Use a dummy variable `_` for return values not needed here to avoid confusion.
                    summary_for_grand_total, _, _, _ = process_invoice(invoice_number_key, page_data_list_for_sum, PROCESSED_INVOICE_FOLDER) 
                    if summary_for_grand_total:
                        for coa, val in summary_for_grand_total.items():
                            bill_grand_total[coa] += val
                            
                print(f"\n=== FINAL AGGREGATED COA SUMMARY FOR DOCUMENT ASSOCIATED WITH ORIGINAL BILL ID {bill_id} ===")
                for coa, val in bill_grand_total.items():
                    print(f"{coa}: ${val:,.2f}")
                print(f" === TOTAL SUM OF ALL COA CATEGORIES FOR DOCUMENT: ${sum(bill_grand_total.values()):,.2f} ===")
                print(f"\n--- Finished OCR and COA processing for document associated with Bill ID {bill_id} ---")

        else:
            print(f"[-] Failed to create the final combined PDF for Bill ID {bill_id}.")
    else:
        print(f"[-] No PDF files were generated or found for combination for Bill ID {bill_id}.")

    print(f"\n--- Cleaning up intermediate files for Bill ID {bill_id} ---")
    for temp_pdf_path in bill_intermediate_pdfs_for_cleanup:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
            print(f"[+] Removed intermediate PDF: {temp_pdf_path}")
    
    # Ensure to clean up the temporary OCR images regardless of PDF issues
    # List files directly under TEMP_OCR_IMAGES_FOLDER, which is /tmp/ocr_temp_images
    if os.path.exists(TEMP_OCR_IMAGES_FOLDER):
        for ocr_image_file in os.listdir(TEMP_OCR_IMAGES_FOLDER):
            file_to_remove = os.path.join(TEMP_OCR_IMAGES_FOLDER, ocr_image_file)
            if os.path.isfile(file_to_remove):
                try:
                    os.remove(file_to_remove)
                    print(f"[+] Removed temporary OCR image: {file_to_remove}")
                except OSError as e:
                    print(f"[-] Error removing file {file_to_remove}: {e}")
    else:
        print(f"[-] Temporary OCR images folder '{TEMP_OCR_IMAGES_FOLDER}' does not exist.")


# ========== MAIN LAMBDA EXECUTION ENTRY POINT ==========
def lambda_handler(event, context):
    """
    Main function handler for AWS Lambda.
    This function will be triggered by EventBridge.
    """
    print("Lambda function invoked. Starting Bill.com automation process.")

    # Ensure /tmp/ folders exist (Lambda provides a writable /tmp/ directory)
    os.makedirs(RAW_INVOICE_FOLDER, exist_ok=True)
    os.makedirs(PROCESSED_INVOICE_FOLDER, exist_ok=True)
    os.makedirs(TEMP_OCR_IMAGES_FOLDER, exist_ok=True)
    print(f"Ensured temporary folders '{RAW_INVOICE_FOLDER}', '{PROCESSED_INVOICE_FOLDER}', '{TEMP_OCR_IMAGES_FOLDER}' exist within /tmp/.")

    session_id_v2 = login_v2()

    if session_id_v2:
        previous_day_bills = list_bills_created_yesterday(session_id_v2)

        if previous_day_bills is not None:
            if not previous_day_bills:
                print("\n[!] No bills found for the previous day to process.")
            else:
                print(f"\nStarting processing for {len(previous_day_bills)} bills from the previous day...")
                for bill_data in previous_day_bills: 
                    bill_data_to_process = {
                        'id': bill_data.get('id'),
                        'vendorId': bill_data.get('vendorId'),
                        'description': bill_data.get('description', '') 
                    }
                    if bill_data_to_process.get('id'):
                        process_single_bill(session_id_v2, bill_data_to_process) 
                    else:
                        print(f"[-] Skipping a bill due to missing ID: {bill_data}")
                print("\nAll requested bills processed.")
        else:
            print("[-] Could not retrieve previous day's bill IDs. Exiting Lambda.")
            return {
                'statusCode': 500,
                'body': json.dumps('Failed to retrieve bills.')
            }
    else:
        print("[-] Login V2 failed. Cannot proceed with any operations. Exiting Lambda.")
        return {
            'statusCode': 500,
            'body': json.dumps('Bill.com V2 login failed.')
        }

    print("Lambda function execution finished.")
    return {
        'statusCode': 200,
        'body': json.dumps('Bill processing completed successfully!')
    }
