# BERT & IMDb fine-tuning

transformers Library의 BERT를 활용하여 model 작동 방식 탐색 및 fine-tuning 예제

### 출처

[딥 러닝을 이용한 자연어 처리 입문](https://wikidocs.net/24586)\
[Getting Started with Google BERT, 구글 BERT의 정석](https://github.com/PacktPublishing/Getting-Started-with-Google-BERT?tab=readme-ov-file)

## 프로젝트 소개

BERT pretrain을 불러온 뒤 model의 구성 요소 및 config를 확인한다. \
tokenizer의 작동 방식을 파악한다. \
model의 output이 정확히 어떤 것을 의미하는지 살펴본다. \ 
IMDb dataset의 구조를 파악하고 전처리를 진행한다. \ 
trainer를 활용해 model 전체를 학습시킨다. \
pytorch만을 활용하여 BERT 뒤에 linear layer를 연결한 후 이 layer만 학습시킨다. 


## 실행 환경
ipynb 파일 Colab에서 실행. \
다른 라이브러리 설치나 파일 필요 없음.