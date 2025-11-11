import socket
import urllib.request
import urllib.error
from typing import Optional, List, Tuple


class InternetChecker:
    """
    A robust class to check internet connectivity using multiple methods.
    """

    def __init__(self, timeout: float = 5.0):
        """
        Initialize the checker.
        
        :param timeout: Timeout in seconds for all connection attempts.
        """
        self.timeout = timeout
        
        # Multiple fallback hosts and ports for socket-based checking
        self.socket_hosts = [
            ("8.8.8.8", 53),      # Google DNS
            ("1.1.1.1", 53),      # Cloudflare DNS
            ("208.67.222.222", 53),  # OpenDNS
            ("8.8.4.4", 53),      # Google DNS Secondary
        ]
        
        # URLs for HTTP-based checking
        self.http_urls = [
            "http://www.google.com",
            "http://www.cloudflare.com",
            "http://www.github.com",
            "http://httpbin.org/status/200"
        ]

    def _check_socket_connection(self, host: str, port: int) -> bool:
        """
        Check connectivity using socket connection.
        
        :param host: Host to connect to.
        :param port: Port to connect to.
        :return: True if connection successful, False otherwise.
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(self.timeout)
                result = sock.connect_ex((host, port))
                return result == 0
        except (socket.timeout, socket.error, OSError):
            return False

    def _check_http_connection(self, url: str) -> bool:
        """
        Check connectivity using HTTP request.
        
        :param url: URL to test.
        :return: True if request successful, False otherwise.
        """
        try:
            request = urllib.request.Request(url)
            request.add_header('User-Agent', 'Mozilla/5.0 (compatible; InternetChecker/1.0)')
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.getcode() == 200
        except (urllib.error.URLError, urllib.error.HTTPError, socket.timeout, OSError):
            return False

    def is_connected(self) -> bool:
        """
        Check if the internet is connected using multiple methods.
        
        :return: True if connected, False otherwise.
        """
        # First, try socket-based connections (faster and more reliable)
        for host, port in self.socket_hosts:
            if self._check_socket_connection(host, port):
                return True
        
        # If socket connections fail, try HTTP-based connections
        for url in self.http_urls[:2]:  # Only try first 2 URLs to avoid delays
            if self._check_http_connection(url):
                return True
        
        return False

    def get_detailed_status(self) -> dict:
        """
        Get detailed connectivity status with test results.
        
        :return: Dictionary with detailed status information.
        """
        socket_results = []
        http_results = []
        
        # Test socket connections
        for host, port in self.socket_hosts:
            result = self._check_socket_connection(host, port)
            socket_results.append({
                "host": f"{host}:{port}",
                "connected": result
            })
        
        # Test HTTP connections
        for url in self.http_urls:
            result = self._check_http_connection(url)
            http_results.append({
                "url": url,
                "connected": result
            })
        
        overall_connected = any(r["connected"] for r in socket_results) or any(r["connected"] for r in http_results)
        
        return {
            "overall_connected": overall_connected,
            "socket_tests": socket_results,
            "http_tests": http_results,
            "timeout_used": self.timeout
        }

    def get_status_message(self) -> str:
        """
        Get a human-readable status message.
        
        :return: Status string.
        """
        return "Internet is connected" if self.is_connected() else "No internet connection"

    def test_custom_server(self, host: str, port: int, timeout: Optional[float] = None) -> bool:
        """
        Test connectivity to a custom server.
        
        :param host: Host to test.
        :param port: Port to test.
        :param timeout: Optional timeout override.
        :return: True if reachable, False otherwise.
        """
        test_timeout = timeout if timeout is not None else self.timeout
        return self._check_socket_connection(host, port)

    def test_custom_url(self, url: str, timeout: Optional[float] = None) -> bool:
        """
        Test connectivity to a custom URL.
        
        :param url: URL to test.
        :param timeout: Optional timeout override.
        :return: True if reachable, False otherwise.
        """
        if timeout is not None:
            original_timeout = self.timeout
            self.timeout = timeout
            result = self._check_http_connection(url)
            self.timeout = original_timeout
            return result
        return self._check_http_connection(url)


def quick_internet_check() -> bool:
    """
    Quick function to check internet connectivity.
    
    :return: True if connected, False otherwise.
    """
    checker = InternetChecker(timeout=3.0)
    return checker.is_connected()


if __name__ == "__main__":
    # Basic testing
    checker = InternetChecker()
    print("="*50)
    print("BASIC CONNECTIVITY CHECK")
    print("="*50)
    
    print(f"Status: {checker.get_status_message()}")
    print(f"Connected: {checker.is_connected()}")
    
    print("\n" + "="*50)
    print("DETAILED CONNECTIVITY STATUS")
    print("="*50)
    
    detailed_status = checker.get_detailed_status()
    print(f"Overall Connected: {detailed_status['overall_connected']}")
    print(f"Timeout Used: {detailed_status['timeout_used']}s")
    
    print("\nSocket Tests:")
    for test in detailed_status['socket_tests']:
        status = "✓" if test['connected'] else "✗"
        print(f"  {status} {test['host']}")
    
    print("\nHTTP Tests:")
    for test in detailed_status['http_tests']:
        status = "✓" if test['connected'] else "✗"
        print(f"  {status} {test['url']}")
    
    print("\n" + "="*50)
    print("CUSTOM SERVER TESTS")
    print("="*50)
    
    # Test custom servers
    custom_tests = [
        ("www.google.com", 80),
        ("www.github.com", 443),
        ("1.1.1.1", 53)
    ]
    
    for host, port in custom_tests:
        result = checker.test_custom_server(host, port)
        status = "✓" if result else "✗"
        print(f"  {status} {host}:{port}")
    
    print("\n" + "="*50)
    print("QUICK CHECK FUNCTION")
    print("="*50)
    print(f"Quick check result: {quick_internet_check()}")