import time
import requests


RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def request_with_retry(
    url,
    *,
    params=None,
    headers=None,
    timeout=30,
    max_retries=3,
    status_codes_to_retry=None,
    session=None,
):
    """
    Send a GET request with controlled retry/backoff for transient failures.
    """
    if status_codes_to_retry is None:
        status_codes_to_retry = RETRYABLE_STATUS_CODES

    client = session or requests
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            response = client.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout,
            )

            if response.status_code in status_codes_to_retry and attempt < max_retries:
                retry_after = response.headers.get("Retry-After")
                if retry_after is not None:
                    try:
                        sleep_seconds = max(float(retry_after), 0.0)
                    except ValueError:
                        sleep_seconds = 0.5 * (2 ** attempt)
                else:
                    sleep_seconds = 0.5 * (2 ** attempt)

                time.sleep(sleep_seconds)
                continue

            return response

        except (requests.Timeout, requests.ConnectionError, requests.RequestException) as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(0.5 * (2 ** attempt))
                continue
            raise

    if last_error is not None:
        raise last_error

    raise requests.RequestException("Request failed without a response")