from DIPLOMA.services.request_service import RequestService


class CoinMarketCapService(RequestService):
    token = 'fc272bd5-4d50-440f-b0fb-5b93f40c1ebf'
    host_domain = 'https://pro-api.coinmarketcap.com'
    authorization_header_name = 'X-CMC_PRO_API_KEY'

    @classmethod
    def _generate_token(cls) -> str:
        return cls.token

    @classmethod
    def listings(cls):
        return cls.get('v1/cryptocurrency/listings/latest')
