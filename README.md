1. main.py 
 > data에 존재하는 파일을 읽어와 학습을 진행함

 data 구조
 data
   |
   --CamVid
        |
        -- Train >> 학습용 데이터셋 원본 이미지
        |
        -- Train_label >> 학습용 데이터셋 라벨 이미지
        |
        -- Test >> 테스트용 데이터셋 원본 이미지
        |
        -- Test_label >> 테스트용 데이터셋 라벨 이미지
        |
        -- Val >> 검증용 데이터셋 원본 이미지
        |
        -- Val_label >> 검증 데이터셋 라벨 이미지
        |
        -- class_dict.csv >> 각 Mask의 색상별 객체 라벨링이 되어있는 csv파일


pt 파일은 모델을 저장하고
pth 파일은 모델의 가중치를 저장한다

모델을 로드할땐 torch.load("model.pt")해서 모델을 로드하고
LoadCamVid, Preprocessing, evaluation 세가지의 파일에서 함수로 불러와 입력데이터를 전처리하여
모델에 넣고 돌리면 결과가 나온다.