from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


def _extract_urls_helper(url, visited, original_url, max_depth=1, current_depth=1):
    """
    Extracts all URLs from a given URL, recursively up to a maximum depth.
    """
    if (
        url in visited
        or urlparse(url).netloc != urlparse(original_url).netloc
        or "redirect" in url
    ):
        return []

    visited.add(url)

    urls = [url]

    if current_depth <= max_depth:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        for link in soup.find_all("a"):
            href = link.get("href")
            if href:
                # Ignore links within the same page
                if href.startswith("#"):
                    continue

                if not href.startswith("http"):
                    href = urljoin(url, href)

                parsed_href = urlparse(href)
                if bool(parsed_href.scheme) and bool(parsed_href.netloc):
                    urls.extend(
                        _extract_urls_helper(
                            href, visited, original_url, max_depth, current_depth + 1
                        )
                    )
    return urls


def extract_urls(url, max_depth=1):
    visited = set()
    return _extract_urls_helper(
        url, visited, original_url=url, max_depth=max_depth, current_depth=1
    )
