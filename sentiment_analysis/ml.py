import fasttext
import os
hyper_params = {"lr": 0.225,
    "epoch": 5000,
    "wordNgrams": 2,
    "dim": 20}  
model = fasttext.train_supervised(input="train",**hyper_params)
#results
result=model.test("train")
validation=model.test("test")
# DISPLAY ACCURACY OF TRAINED MODEL
text_line = str(hyper_params) + ",accuracy:" + str(result[1])  + ",validation:" + str(validation[1]) + '\n' 
print(text_line)

f_data= open("results", 'a+') 
f_data.write(text_line)
f_data.close()

model.save_model("model.bin")

# print(model.predict("your word "))

