import pandas
import re
from dotenv import load_dotenv
from google import genai
import os
from google.genai import types
import httpx
from google.genai.types import GenerateContentConfig
import threading
import concurrent.futures

from constants import *
import json
import io
import pikepdf
import time
from collections import deque
from datetime import datetime, timedelta


def initialize_clients() -> list:
    """
    Creates multiple Gemini clients using available API keys from .env

    :return: List of genai.Client objects
    """
    load_dotenv()
    clients = []

    # Get all API keys (current and potential future ones)
    api_keys = []
    key_index = 1
    while True:
        # Try to get API keys with different naming patterns
        if key_index == 1:
            key = os.getenv("GOOGLE_API")
        else:
            key = os.getenv(f"GOOGLE_API_{key_index}")

        if not key:
            break

        api_keys.append(key)
        key_index += 1

    # Also check for numbered keys starting from TWO for backward compatibility
    if os.getenv("GOOGLE_API_TWO"):
        api_keys.append(os.getenv("GOOGLE_API_TWO"))
    if os.getenv("GOOGLE_API_THREE"):
        api_keys.append(os.getenv("GOOGLE_API_THREE"))

    # Remove duplicates while maintaining order
    unique_keys = []
    for key in api_keys:
        if key not in unique_keys:
            unique_keys.append(key)

    # Create clients for each unique key
    for key in unique_keys:
        clients.append(genai.Client(api_key=key))

    print(f"Initialized {len(clients)} API clients")
    return clients


class ThreadSafeRateLimiter:
    """
    Thread-safe rate limiter for the Gemini API that enforces limits of:
    - 15 requests per minute
    - 1500 requests per day
    Per API key
    """

    def __init__(self, requests_per_minute=15, requests_per_day=1500):
        self.requests_per_minute = requests_per_minute
        self.requests_per_day = requests_per_day
        self.minute_requests = deque()
        self.day_requests = deque()
        self.lock = threading.Lock()
        print(
            f"Rate limiter initialized with {requests_per_minute} requests/minute and {requests_per_day} requests/day")

    def can_make_request(self):
        """Check if a request can be made based on rate limits"""
        with self.lock:
            current_time = datetime.now()

            # Clean up expired timestamps
            self._cleanup_timestamps(current_time)

            # Check if we're within the limits
            minute_ok = len(self.minute_requests) < self.requests_per_minute
            day_ok = len(self.day_requests) < self.requests_per_day

            if not minute_ok:
                print(f"Rate limit: Minute limit reached ({len(self.minute_requests)}/{self.requests_per_minute})")
            if not day_ok:
                print(f"Rate limit: Daily limit reached ({len(self.day_requests)}/{self.requests_per_day})")

            return minute_ok and day_ok

    def record_request(self):
        """Record a request timestamp"""
        with self.lock:
            current_time = datetime.now()
            self.minute_requests.append(current_time)
            self.day_requests.append(current_time)
            print(
                f"Request recorded. Minute: {len(self.minute_requests)}/{self.requests_per_minute}, Day: {len(self.day_requests)}/{self.requests_per_day}")

    def wait_until_allowed(self):
        """Wait until a request is allowed based on rate limits"""
        while True:
            with self.lock:
                if self.can_make_request():
                    return

            # Release lock while waiting to prevent deadlock
            print("Rate limit reached. Waiting for next available slot...")
            time.sleep(1)

    def _cleanup_timestamps(self, current_time):
        """Remove timestamps that are outside the time windows"""
        # Remove requests older than 1 minute
        minute_cutoff = current_time - timedelta(minutes=1)
        while self.minute_requests and self.minute_requests[0] < minute_cutoff:
            self.minute_requests.popleft()

        # Remove requests older than 1 day
        day_cutoff = current_time - timedelta(days=1)
        while self.day_requests and self.day_requests[0] < day_cutoff:
            self.day_requests.popleft()


class ClientManager:
    """
    Manages a pool of API clients with their own rate limiters
    """

    def __init__(self, clients):
        self.clients = clients
        self.rate_limiters = [ThreadSafeRateLimiter() for _ in clients]
        self.current_index = 0
        self.lock = threading.Lock()

    def get_next_available_client(self):
        """
        Returns the next available client that can make a request
        """
        wait_reported = False

        while True:
            with self.lock:
                # Check all clients in a round-robin fashion
                for _ in range(len(self.clients)):
                    idx = self.current_index
                    self.current_index = (self.current_index + 1) % len(self.clients)

                    if self.rate_limiters[idx].can_make_request():
                        return self.clients[idx], self.rate_limiters[idx], idx

            # If we've checked all clients and none are available, wait
            if not wait_reported:
                print("All API keys are rate limited. Waiting...")
                wait_reported = True
            time.sleep(1)


class ExcelManager:
    """
    Thread-safe manager for Excel file operations
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.lock = threading.Lock()

    def append_dataframe(self, df):
        """
        Appends a dataframe to the Excel file in a thread-safe manner

        :param df: pandas DataFrame containing data to append
        :return: None
        """
        if df is None:
            return

        with self.lock:
            try:
                # Check if file exists
                if os.path.exists(self.file_path):
                    # Read existing file
                    existing_df = pandas.read_excel(self.file_path)
                    # Append the new dataframe
                    updated_df = pandas.concat([existing_df, df], ignore_index=True)
                    # Write back to file
                    updated_df.to_excel(self.file_path, index=False)
                else:
                    # Create new file if it doesn't exist
                    df.to_excel(self.file_path, index=False)

                print(f"Data successfully written to {self.file_path}")
            except Exception as e:
                print(f"Error writing to Excel file: {e}")


def process_company(client_manager, excel_manager, row_data, row_i, total_rows, model_name, system_instructions):
    """
    Process a single company using available API client

    :param client_manager: ClientManager instance
    :param excel_manager: ExcelManager instance
    :param row_data: DataFrame row containing company data
    :param row_i: Row index
    :param total_rows: Total number of rows
    :param model_name: Model name to use
    :param system_instructions: System instructions for LLM
    :return: True if processed successfully, False otherwise
    """
    pitch = row_data["Attachments"]
    if pitch == "":
        return False

    pdf_string: str = str(pitch)
    pattern = r"https?://[^\)]+"
    links = re.findall(pattern, pdf_string)
    HiCap = row_data["HiCap"]

    if not links:
        return False

    doc_url = links[len(links) - 1]

    # Retrieve PDF data
    try:
        doc_data = httpx.get(doc_url).content
    except Exception as e:
        print(f"Error downloading PDF for {HiCap}: {e}")
        return False

    prompt = pdf_prompt

    # Get next available client
    client, rate_limiter, client_idx = client_manager.get_next_available_client()

    print(f"Processing company {row_i + 1}/{total_rows}: {HiCap} using client {client_idx + 1}")

    try:
        # Make the request
        response = client.models.generate_content(
            model=model_name,
            contents=[
                types.Part.from_bytes(
                    data=doc_data,
                    mime_type='application/pdf',
                ),
                prompt],
            config=GenerateContentConfig(
                system_instruction=[
                    system_instructions,
                ]
            )
        )

        # Record the request
        rate_limiter.record_request()

        # Process response
        response_df = llm_to_df(response.text, HiCap)

        # Write to Excel in a thread-safe way
        excel_manager.append_dataframe(response_df)

        return True
    except Exception as e:
        print(f"Error processing company {HiCap}: {e}")
        return False


def main(model_name: str = "gemini-2.0-flash", num_companies: int = None,
         csv_file_path: str = "airtable-data/Deals-Pipeline View Clean.csv",
         company_name: str = None, file_path: str = "output.xlsx",
         system_instructions: str = "", max_workers: int = None):
    """
    This function runs the system with multiple threads and API keys.

    :param model_name: str
    :param num_companies: int
    :param csv_file_path: str
    :param company_name: str - If provided, processing starts from this company
    :param file_path: str
    :param system_instructions: str
    :param max_workers: Maximum number of worker threads (defaults to number of API keys)
    :return: None
    """
    # Initialize clients and managers
    clients = initialize_clients()
    client_manager = ClientManager(clients)
    excel_manager = ExcelManager(file_path)

    # If max_workers not specified, use number of clients
    if max_workers is None:
        max_workers = len(clients)

    with open(csv_file_path) as airtable_f:
        df = pandas.read_csv(airtable_f)

    total_rows = df.shape[0]

    # Determine starting index based on company_name
    start_index = 0
    if company_name is not None:
        company_rows = df[df["HiCap"] == company_name].index.tolist()
        if company_rows:
            start_index = company_rows[0]
            print(f"Starting processing from company '{company_name}' at index {start_index}")
        else:
            print(f"Company '{company_name}' not found in dataset. Starting from the beginning.")

    # Create a queue of companies to process
    companies_to_process = []
    for row_i in range(start_index, total_rows):
        companies_to_process.append((df.iloc[row_i], row_i))
        if num_companies is not None and len(companies_to_process) >= num_companies:
            break

    processed_count = 0
    total_to_process = len(companies_to_process)

    print(f"Starting processing of {total_to_process} companies with {max_workers} worker threads")

    # Use ThreadPoolExecutor to process companies in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a dict to track futures
        future_to_company = {
            executor.submit(
                process_company,
                client_manager,
                excel_manager,
                company_data,
                row_i,
                total_rows,
                model_name,
                system_instructions
            ): (company_data, row_i)
            for company_data, row_i in companies_to_process
        }

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_company):
            company_data, row_i = future_to_company[future]
            try:
                success = future.result()
                if success:
                    processed_count += 1
                print(f"Completed {processed_count}/{total_to_process} companies")
            except Exception as e:
                print(f"Error processing company at row {row_i + 1}: {e}")

    print(f"Completed processing {processed_count} companies with multithreading")

def llm_to_df(llm_output: str, company_name: str) -> pandas.DataFrame | None:
    """
    Converts LLM output in JSON format to a pandas DataFrame

    :param llm_output: str
    :param company_name: str
    :return: pandas.DataFrame | None
    """
    if not llm_output:
        return None

    # Clean up the output by removing markdown code blocks
    clean_output = llm_output.replace("```json", "").replace("```", "").strip()

    try:
        # Parse the JSON
        json_data = json.loads(clean_output)
        company_dict = {"company_name": company_name}
        json_data = dict(company_dict, **json_data)
        # If json_data is a dictionary, wrap it in a list to create a single-row DataFrame
        if isinstance(json_data, dict):
            return pandas.DataFrame([json_data])
        # If json_data is already a list, create DataFrame directly
        elif isinstance(json_data, list):
            return pandas.DataFrame(json_data)
        else:
            print(f"Unexpected JSON structure: {type(json_data)}")
            return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None