import requests
import os

# Create output folder
os.makedirs("bmc_articles", exist_ok=True)

# Europe PMC API (query for "BMC Medicine")
api_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

params = {
    "query": 'JOURNAL:"BMC Medicine"',   # restrict to BMC Medicine
    "resultType": "core",
    "format": "json",
    "pageSize": 100
       # number of articles to fetch (increase if needed)
}

response = requests.get(api_url, params=params).json()

count = 0
for article in response.get("resultList", {}).get("result", []):
    title = article.get("title", "No title")
    pmcid = article.get("pmcid")

    if not pmcid:
        continue

    # Fetch full text in plain text
    fulltext_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
    r = requests.get(fulltext_url)

    if r.status_code == 200:
        filename = f"bmc_articles/article_{count}.md"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n")
            f.write(r.text)  # XML format, can be cleaned later
        print(f"✅ Saved {filename}")
        count += 1
