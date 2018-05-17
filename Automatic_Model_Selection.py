# -*- coding: utf-8 -*-

# Title          : Automatic Model Selection
# Author         : 김은종
# Origin_Date    : 05/11/2018
# Revision_Date  : 05/17/2018
# Version        : '0.1.4'


import h2o
from h2o.automl import H2OAutoML

h2o.init()

# load_file Definition Example
df = h2o.import_file(path="./Test/creditcard.csv", destination_frame="df")
print(df.head())

# Input parameters that are going to train (Target)
# response_column = 'Amount'
response_column = 'Class'

print(response_column)

training_columns = df.columns.remove(response_column)
print(training_columns)
# Output parameter train against input parameters
# TODO: from h2o.estimators.deeplearning import H2OAutoEncoderEstimator (H2OAutoEncoder)

# Split data into train and testing
train, test = df.split_frame(ratios=[0.7])

# For regression     --- default
# For classification --- binary

while True:
    try:
        response_type = input("변수 형태를 입력해주세요. (default/binary): ")
        break
    except TypeError:
        print("입력한 값은 문자열이 아닙니다. 다시 입력해주세요,")

# For binary classification, response should be a factor
if response_type == 'binary':
    train[response_column] = train[response_column].asfactor()
    test[response_column] = test[response_column].asfactor()

### AutoML

# Time to run the experiment
while True:
    try:
        run_automl_time = int(input("최대 허용 시간을 입력해주세요. (s): "))
        break
    except ValueError:
        print("입력한 값은 정수가 아닙니다. 다시 입력해주세요.")

# RUN AutoML
aml = H2OAutoML(max_runtime_secs=run_automl_time)

aml.train(x=training_columns, y=response_column,
          training_frame=train,
          leaderboard_frame=test)

# View the AutoML Leaderboard

lb = aml.leaderboard

# Get Model Ids
model_ids = list(lb['model_id'].as_data_frame().iloc[:, 0])
model_number = h2o.H2OFrame(list(range(1,len(model_ids)+1)),column_names=[' '])

print(model_number.cbind(lb))

# Model Selection (input)
while True:
    try:
        model_number = int(input("원하는 모델 번호를 입력하세요.: "))
        selected_model = h2o.get_model(model_ids[model_number-1])
        print(selected_model)
        break
    except ValueError:
        print("입력한 값은 정수가 아닙니다. 다시 입력해주세요.")

while True:
    key_input = input("예측 모델과 결과 값을 저장하시겠습니까? (y/n): ")

    if key_input.upper() == 'Y':
        pred = selected_model.predict(test_data=test)
        pred_df = pred.as_data_frame()
        pred_df.to_csv('./Test/creditcard_result.csv', header=True, index=False)
        h2o.save_model(selected_model, path="./product_backorders_model_bin")
        print("저장되었습니다.")
        break
    elif key_input.upper() == 'N':
        print("저장되지 않았습니다.")
        break
    else:
        print("잘못 입력하셨습니다. 다시 입력해주세요.")