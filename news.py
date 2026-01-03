print('hiii test')
print('hi')

!pip install selenium webdriver-manager

import time
import pandas as pd
from bs4 import BeautifulSoup as bs
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# 1. 수집하고 싶은 카테고리를 여기에 적으세요 (정치, 경제, 사회, 문화 등)
target_category = "사회" 

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
wait = WebDriverWait(driver, 15)
news_data = []

try:
    driver.get("https://www.bigkinds.or.kr/v2/news/recentNews.do")
    
    # 2. 카테고리 버튼 클릭 (target_category 변수 활용)
    # 이미지 image_4cc661.jpg 구조 반영
    category_xpath = f"//button[contains(., '{target_category}')]"
    category_btn = wait.until(EC.element_to_be_clickable((By.XPATH, category_xpath)))
    category_btn.click()
    time.sleep(3) 

    while len(news_data) < 100:
        soup = bs(driver.page_source, 'html.parser')
        items = soup.select("div.news-item")
        
        for item in items:
            if len(news_data) >= 100: break
                
            try:
                # [제목] strong.title (image_418403.jpg 확인)
                title = item.select_one("strong.title").get_text(strip=True)
                
                # [신문사] a.provider (image_4c55e8.jpg 확인)
                press_el = item.select_one("a.provider")
                press = press_el.get_text(strip=True) if press_el else "신문사 미상"
                
                # [내용] p.text (image_4c5626.jpg 확인)
                content_el = item.select_one("p.text")
                content = content_el.get_text(strip=True) if content_el else "내용 없음"
                
                if title not in [n['제목'] for n in news_data]:
                    news_data.append({
                        '번호': len(news_data) + 1,
                        '제목': title,
                        '신문사': press,
                        '내용': content
                    })
            except Exception:
                continue

        print(f"[{target_category}] 현재 {len(news_data)}개 수집 중...")

        if len(news_data) >= 100: break

        # 3. 다음 페이지 클릭 (image_41e0dd.png 확인)
        try:
            next_btn = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "page-next")))
            driver.execute_script("arguments[0].click();", next_btn)
            time.sleep(3) 
        except:
            break

    # 4. 저장 파일명에 카테고리 이름 포함
    df = pd.DataFrame(news_data)
    df = df[['번호', '제목', '신문사', '내용']]
    df.to_csv(f"bigkinds_{target_category}_100.csv", index=False, encoding='utf-8-sig')
    print(f"✅ {target_category} 섹션 100개 저장 완료!")

except Exception as e:
    print(f"오류 발생: {e}")

finally:
    driver.quit()