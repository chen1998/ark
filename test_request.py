import requests

if __name__ == "__main__":
    # 目標URL
    target_url = "http://tcweb002.corpnet.auo.com/AAMEL310/Results/"
    
    # 代理伺服器設定
    proxies = {
        'http': 'http://10.97.4.1:8080',
        'https': 'http://10.97.4.1:8080'
    }
    
    # 設定模擬瀏覽器的 header
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0'
    }
    
    # 發送GET請求，加入 headers 參數
    response = requests.get(target_url, proxies=proxies, headers=headers)
    
    response.encoding = 'utf8'
    print(response.text)