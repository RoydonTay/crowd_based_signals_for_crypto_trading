import requests
import os

from dotenv import load_dotenv

load_dotenv()
client_id = os.getenv('REDDIT_CLIENT_ID')
client_secret = os.getenv('REDDIT_CLIENT_SECRET')
password = os.getenv('REDDIT_PASSWORD')

assert client_id, "REDDIT_CLIENT_ID is not set"
assert client_secret, "REDDIT_CLIENT_SECRET is not set"
assert password, "REDDIT_PASSWORD is not set"

auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
headers = {'User-Agent': 'MyPythonScript/1.0 (Custom User-Agent)'}
data = {
    'grant_type' : 'password',
    'username' : 'Basic-Newspaper',
    'password' : password
}

try:
    response = requests.post("https://www.reddit.com/api/v1/access_token", auth=auth, data=data, headers=headers)
    
    print(f"Response status: {response.status_code}")
    print(f"Response headers: {dict(response.headers)}")
    print(f"Response text: {response.text}")
    print()
    
    response.raise_for_status()  # Raises an HTTPError for bad responses
    
    # Extract access token
    token = response.json()['access_token']
    print(f"Successfully obtained access token!")
    
    # Update headers with authorization
    api_headers = {
        'User-Agent': 'MyPythonScript/1.0 (Custom User-Agent)',
        'Authorization': f'bearer {token}'
    }
    
    # Reddit Search API call
    # Search parameters (you can modify these)
    search_params = {
        'q': '12 August 2025',           # Search query (max 512 characters)
        'limit': 10,                        # Number of results (default: 25, max: 100)
        'sort': 'relevance',               # Sort by: relevance, hot, top, new, comments
        't': 'week',                       # Time period: hour, day, week, month, year, all
        'type': 'link',                    # Result types: sr, link, user (comma-delimited)
        'restrict_sr': 'true',             # Restrict search to specified subreddit
        'include_facets': 'false',         # Include facets in response
        'show': 'all',                     # Show all results
        'count': 0,                        # Starting count (default: 0)
        # 'after': '',                     # fullname of a thing (for pagination)
        # 'before': '',                    # fullname of a thing (for pagination)
        # 'category': '',                  # Category filter (max 5 characters)
        # 'sr_detail': 'false'             # Expand subreddit details
    }
    
    # Subreddit to search in (change this to your desired subreddit)
    subreddit = 'help'
    
    # Make the search API call
    search_url = f"https://oauth.reddit.com/r/{subreddit}/search"
    
    print(f"\nMaking search request to: {search_url}")
    print(f"Search parameters: {search_params}")
    
    search_response = requests.get(search_url, headers=api_headers, params=search_params)
    
    print(f"\nSearch Response status: {search_response.status_code}")
    print(f"Search Response headers: {dict(search_response.headers)}")
    
    if search_response.status_code == 200:
        search_data = search_response.json()
        print(f"\nSearch successful!")
        print(f"Number of results: {len(search_data.get('data', {}).get('children', []))}")
        
        # Display first few results
        posts = search_data.get('data', {}).get('children', [])
        for i, post in enumerate(posts[:3]):  # Show first 3 results
            post_data = post.get('data', {})
            print(f"\nResult {i+1}:")
            print(post_data)
            # print(f"  Title: {post_data.get('title', 'N/A')}")
            # print(f"  Subreddit: r/{post_data.get('subreddit', 'N/A')}")
            # print(f"  Score: {post_data.get('score', 'N/A')}")
            # print(f"  URL: {post_data.get('url', 'N/A')}")
            # print(f"  Created: {post_data.get('created_utc', 'N/A')}")
    else:
        print(f"Search failed with status: {search_response.status_code}")
        print(f"Error response: {search_response.text}")

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
except KeyError as e:
    print(f"Error parsing response: {e}")
    print("This might indicate authentication failed or response format changed")