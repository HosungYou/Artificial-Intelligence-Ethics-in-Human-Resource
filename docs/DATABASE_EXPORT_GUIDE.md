# Database Export Guide
## AI Ethics in HR Systematic Review

이 문서는 각 기관 데이터베이스에서 검색 결과를 내보내는 방법을 설명합니다.

---

## 검색 쿼리 (모든 데이터베이스 공통)

### 기본 검색어 (HRD 용어 포함)

```
(("artificial intelligence" OR "AI" OR "machine learning" OR "algorithm*" OR
  "automated" OR "chatbot" OR "NLP" OR "predictive analytics")
 AND
 ("human resource*" OR "HR" OR "HRM" OR "HRD" OR "human resource development" OR
  "talent management" OR "recruitment" OR "selection" OR "hiring" OR
  "performance management" OR "learning and development" OR "training and development" OR
  "employee development" OR "workforce analytics" OR "people analytics" OR
  "organizational development" OR "career development" OR "workplace learning")
 AND
 ("ethic*" OR "bias" OR "fairness" OR "discrimination" OR "transparency" OR
  "accountability" OR "privacy" OR "surveillance" OR "trust" OR "responsible AI"))
```

### 필터 조건
- **연도**: 2015-2026
- **언어**: English
- **문서 유형**: Journal articles, Conference papers, Book chapters

---

## 1. Scopus 내보내기

### 접속
1. https://www.scopus.com 접속
2. 기관 계정으로 로그인

### 검색 실행
1. "Advanced search" 선택
2. 다음 쿼리 입력:

```
TITLE-ABS-KEY(("artificial intelligence" OR "AI" OR "machine learning" OR "algorithm*")
AND ("human resource*" OR "HR" OR "HRM" OR "recruitment" OR "selection" OR "performance management" OR "people analytics")
AND ("ethic*" OR "bias" OR "fairness" OR "privacy" OR "transparency" OR "accountability"))
AND PUBYEAR > 2014 AND PUBYEAR < 2026
AND LANGUAGE(english)
AND DOCTYPE(ar OR cp OR ch)
```

### 내보내기 절차
1. 검색 결과에서 "Select all" 클릭
2. "Export" 버튼 클릭
3. 설정:
   - **Format**: CSV
   - **Information to export**:
     - ✅ Citation information
     - ✅ Bibliographical information
     - ✅ Abstract & keywords
     - ✅ Include references
4. "Export" 클릭하여 다운로드

### 파일 명명
```
scopus_export_YYYYMMDD.csv
```

### 저장 위치
```
data/01_search_results/institutional/scopus_export_20260202.csv
```

---

## 2. Web of Science 내보내기

### 접속
1. https://www.webofscience.com/wos/woscc/smart-search 접속
2. 기관 계정으로 로그인

### 검색 실행
1. "Smart Search" 또는 "Advanced Search" 선택
2. 다음 쿼리 입력 (**단일 TS= 절로 통합**):

```
TS=(("artificial intelligence" OR "AI" OR "machine learning" OR "algorithm*" OR "automated" OR "chatbot" OR "NLP" OR "predictive analytics") AND ("human resource*" OR "HR" OR "HRM" OR "HRD" OR "human resource development" OR "talent management" OR "recruitment" OR "selection" OR "hiring" OR "performance management" OR "learning and development" OR "training and development" OR "employee development" OR "workforce analytics" OR "people analytics" OR "organizational development" OR "career development" OR "workplace learning") AND ("ethic*" OR "bias" OR "fairness" OR "discrimination" OR "transparency" OR "accountability" OR "privacy" OR "surveillance" OR "trust" OR "responsible AI"))
```

3. 필터 적용:
   - **Publication Years**: 2015-2026
   - **Languages**: English
   - **Document Types**: Article, Proceedings Paper, Book Chapter

### Web of Science Categories (분야 선택)

#### 1순위: HR 및 경영/조직 핵심 분야 (가장 추천)
| Category | 건수 | 설명 |
|----------|------|------|
| Management | 531 | HR 전략, 조직 관리, 리더십 |
| Business | 249 | 기업 경영 관점에서의 AI 도입 |
| Applied Psychology | 119 | 채용, 성과 관리 등 HR 실무 |
| Industrial Relations & Labor | 70 | 고용 관계, 노동 윤리, 노동자 권리 |

#### 2순위: 윤리 및 사회적 영향 분야
| Category | 건수 | 설명 |
|----------|------|------|
| Social Sciences, Interdisciplinary | 185 | 기술의 사회적 영향 (다학제적) |
| Ethics | 93 | AI 윤리, 알고리즘의 도덕적 책임 |
| Social Issues | 65 | AI로 인한 차별, 프라이버시 |

#### 3순위: 교육 및 개발 (HRD 관련)
| Category | 건수 | 설명 |
|----------|------|------|
| Education & Educational Research | 190 | 일터학습, AI 활용 교육 |

### 내보내기 절차
1. 검색 결과에서 "Select All" (페이지당 또는 전체)
2. "Export" 버튼 클릭
3. 설정:
   - **Record Content**: Full Record and Cited References
   - **File Format**: Excel (XLS) 또는 Tab delimited file
4. 다운로드

### 주의사항
- WoS는 한 번에 최대 1,000개까지만 내보내기 가능
- 결과가 많으면 여러 번 나누어 내보내기 필요

### 완료된 내보내기 (2026-02-02)
```
data/01_search_results/savedrecs.xls
data/01_search_results/savedrecs (1).xls
```

### 저장 위치
```
data/01_search_results/wos_export_20260202.xls
```

---

## 3. PubMed/MEDLINE 내보내기

### 접속
1. https://pubmed.ncbi.nlm.nih.gov 접속

### 검색 실행
1. 다음 검색어 입력:

```
("artificial intelligence"[Title/Abstract] OR "machine learning"[Title/Abstract] OR "algorithm"[Title/Abstract])
AND ("human resources"[Title/Abstract] OR "recruitment"[Title/Abstract] OR "hiring"[Title/Abstract] OR "employee"[Title/Abstract] OR "workforce"[Title/Abstract])
AND ("ethics"[Title/Abstract] OR "bias"[Title/Abstract] OR "fairness"[Title/Abstract] OR "privacy"[Title/Abstract])
```

2. 필터 적용:
   - Publication dates: 2015-2025
   - Language: English
   - Article types: Journal Article, Review

### 내보내기 절차

#### 방법 1: CSV (권장)
1. "Save" 버튼 클릭
2. Selection: All results
3. Format: CSV
4. "Create file" 클릭

#### 방법 2: XML (더 상세한 데이터)
1. "Save" 버튼 클릭
2. Selection: All results
3. Format: PubMed
4. "Create file" 클릭

### 파일 명명
```
pubmed_export_YYYYMMDD.csv
pubmed_export_YYYYMMDD.xml
```

### 저장 위치
```
data/01_search_results/institutional/pubmed_export_20260202.csv
```

---

## 4. ERIC 내보내기

### 접속
1. https://eric.ed.gov 접속

### 검색 실행
1. 다음 검색어 입력:

```
("artificial intelligence" OR "machine learning" OR "AI" OR "algorithm")
AND ("human resource" OR "employee" OR "workforce" OR "training" OR "professional development")
AND ("ethics" OR "bias" OR "fairness" OR "privacy")
```

2. 필터 적용:
   - Publication Date: Since 2015
   - Publication Type: Journal Articles, Reports
   - Full Text Availability: 선택 사항

### 내보내기 절차
1. 검색 결과에서 "Select All"
2. "Export" 클릭
3. Format: CSV 선택
4. 다운로드

### 파일 명명
```
eric_export_YYYYMMDD.csv
```

### 저장 위치
```
data/01_search_results/institutional/eric_export_20260202.csv
```

---

## 내보내기 후 처리

### 1. 파일 확인

모든 파일이 다음 위치에 저장되었는지 확인:

```
data/01_search_results/institutional/
├── scopus_export_20260202.csv
├── wos_export_20260202.csv
├── pubmed_export_20260202.csv
└── eric_export_20260202.csv
```

### 2. 통합 스크립트 실행

```bash
cd /Volumes/External\ SSD/Projects/AI-Ethics-HR-Review

# 기관 DB 파일 가져오기
python -c "
from scripts.utils.institutional_import import InstitutionalImporter
importer = InstitutionalImporter()
papers = importer.import_all('data/01_search_results/institutional/')
importer.save_to_json(papers, 'data/01_search_results/institutional_papers.json')
"
```

### 3. 결과 확인

```bash
# 가져온 논문 수 확인
python -c "
import json
with open('data/01_search_results/institutional_papers.json') as f:
    data = json.load(f)
    print(f'Total papers imported: {data[\"total_papers\"]}')

    # 소스별 분포
    sources = {}
    for p in data['papers']:
        src = p['source']
        sources[src] = sources.get(src, 0) + 1

    for src, count in sorted(sources.items()):
        print(f'  {src}: {count}')
"
```

---

## 예상 검색 결과

| Database | Expected Results | Notes |
|----------|------------------|-------|
| Scopus | 500-800 | 가장 광범위한 커버리지 |
| Web of Science | 300-500 | 고품질 저널 중심 |
| PubMed | 50-150 | 의료/건강 분야 중심 |
| ERIC | 100-200 | 교육 분야 중심 |
| **Total (before dedup)** | **950-1,650** | |
| **After deduplication** | **~500-800** | 예상 중복률 30-40% |

---

## 문제 해결

### Q: 검색 결과가 너무 많습니다
- 검색어를 더 구체적으로 수정
- 연도 범위 축소 (예: 2020-2025)
- 문서 유형 필터 강화

### Q: 검색 결과가 너무 적습니다
- 검색어 OR 조건 확장
- 와일드카드(*) 활용
- 동의어 추가

### Q: 내보내기가 실패합니다
- 한 번에 내보내는 수량 줄이기
- 다른 형식(CSV → RIS) 시도
- 브라우저 캐시 삭제 후 재시도

### Q: 파일이 깨져 보입니다
- UTF-8 인코딩으로 열기
- 메모장 대신 Excel이나 VS Code 사용

---

## 체크리스트

- [ ] Scopus 검색 및 내보내기 완료
- [ ] Web of Science 검색 및 내보내기 완료
- [ ] PubMed 검색 및 내보내기 완료
- [ ] ERIC 검색 및 내보내기 완료
- [ ] 모든 파일 `data/01_search_results/institutional/` 저장 확인
- [ ] 통합 스크립트 실행
- [ ] 결과 JSON 파일 생성 확인

---

*Last Updated: 2026-02-02*
