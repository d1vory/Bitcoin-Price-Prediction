import logging
from abc import ABC, abstractmethod
from typing import Optional

import requests
from requests import Response


class RequestService(ABC):
    """
    Service class for sending requests to other services.
    """
    host_domain: str = ''  # https://google.com
    is_authorization_in_headers_needed: bool = True
    authorization_header_name: str = 'Authorization'
    authorization_header_type: str = ''  # for example 'Bearer' for jwt tokens
    success_status_codes = {200, 201, 204}
    validation_error_status_codes = {}  # raise ValidationError with data from response
    logger: logging.Logger = logging.getLogger('django')
    service_name: str = ''

    @classmethod
    def get(
            cls,
            url: str,
            headers: Optional[dict] = None,
            max_retries: int = 6,
            timeout: int = 10,
            response_type: str = 'json',
            **kwargs
    ):
        if max_retries < 0:
            cls._raise_connection_error()
        url = cls._get_url(url=url)
        headers = cls._generate_headers(headers=headers)
        try:
            response = requests.get(url=url, headers=headers, **kwargs, timeout=timeout)
            return cls._post_process_response(response, response_type)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            return cls.get(url=url, headers=headers, max_retries=max_retries - 1, timeout=timeout + 20, **kwargs)

    @classmethod
    def post(
            cls,
            url: str,
            json: dict,
            headers: Optional[dict] = None,
            max_retries: int = 6,
            timeout: int = 10,
            response_type: str = 'json',
            **kwargs
    ):
        if max_retries < 0:
            cls._raise_connection_error()
        url = cls._get_url(url=url)
        headers = cls._generate_headers(headers=headers)
        try:
            response = requests.post(url=url, json=json, headers=headers, **kwargs, timeout=timeout)
            return cls._post_process_response(response, response_type)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            return cls.post(
                url=url,
                json=json,
                headers=headers,
                max_retries=max_retries - 1,
                timeout=timeout + 20,
                **kwargs
            )

    @classmethod
    def put(
            cls,
            url: str,
            json: dict,
            headers: Optional[dict] = None,
            max_retries: int = 6,
            timeout: int = 10,
            response_type: str = 'json',
            **kwargs
    ):
        if max_retries < 0:
            cls._raise_connection_error()
        url = cls._get_url(url=url)
        headers = cls._generate_headers(headers=headers)
        try:
            response = requests.put(url=url, json=json, headers=headers, **kwargs, timeout=timeout)
            return cls._post_process_response(response, response_type)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            return cls.put(
                url=url,
                json=json,
                headers=headers,
                max_retries=max_retries - 1,
                timeout=timeout + 20,
                **kwargs
            )

    @classmethod
    def patch(
            cls,
            url: str,
            json: dict,
            headers: Optional[dict] = None,
            max_retries: int = 6,
            timeout: int = 10,
            response_type: str = 'json',
            **kwargs
    ):
        if max_retries < 0:
            cls._raise_connection_error()
        url = cls._get_url(url=url)
        headers = cls._generate_headers(headers=headers)
        try:
            response = requests.patch(url=url, json=json, headers=headers, **kwargs, timeout=timeout)
            return cls._post_process_response(response, response_type)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            return cls.patch(
                url=url,
                json=json,
                headers=headers,
                max_retries=max_retries - 1,
                timeout=timeout + 20,
                **kwargs
            )

    @classmethod
    def delete(
            cls,
            url: str,
            headers: Optional[dict] = None,
            max_retries: int = 6,
            timeout: int = 10,
            response_type: str = 'json',
            **kwargs
    ):
        if max_retries < 0:
            cls._raise_connection_error()
        url = cls._get_url(url=url)
        headers = cls._generate_headers(headers=headers)
        try:
            response = requests.delete(url=url, headers=headers, **kwargs, timeout=timeout)
            return cls._post_process_response(response, response_type)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            return cls.delete(url=url, headers=headers, max_retries=max_retries - 1, timeout=timeout + 20, **kwargs)

    @classmethod
    def _get_url(cls, url: str):
        if cls.host_domain in url:
            return url
        return f'{cls.host_domain}{url}' if url.startswith('/') else f'{cls.host_domain}/{url}'

    @classmethod
    def _generate_headers(cls, headers: Optional[dict] = None) -> dict:
        if isinstance(headers, dict) and not headers:
            # explicitly set empty headers
            return headers
        if not headers:
            headers = {}
        if not cls.is_authorization_in_headers_needed:
            return headers
        if cls.authorization_header_name not in headers:
            headers[f'{cls.authorization_header_name}'] = (
                f'{cls.authorization_header_type} {cls._generate_token()}' if
                cls.authorization_header_type else
                f'{cls._generate_token()}'
            )
        return headers

    @classmethod
    @abstractmethod
    def _generate_token(cls) -> str:
        return ''

    @staticmethod
    def _raise_connection_error():
        raise RuntimeError('CONNECTIONERROR')

    @classmethod
    def _post_process_response(cls, response: Response, response_type: str):
        if response.status_code in cls.success_status_codes:
            return cls._on_success_response(response, response_type)
        elif response.status_code in cls.validation_error_status_codes:
            return cls._on_validation_error_response(response, response_type)
        else:
            return cls._on_fail_response(response, response_type)

    @classmethod
    def _on_success_response(cls, response: Response, response_type: str) -> dict | bytes:
        if response_type == 'json':
            return response.json()
        else:
            return response.content

    @classmethod
    def _on_fail_response(cls, response: Response, response_type: str):
        raise RuntimeError(f'{cls.service_name} service responded with error {response.status_code}!')

    @classmethod
    def _on_validation_error_response(cls, response, response_type: str):
        cls._on_fail_response(response, response_type)

