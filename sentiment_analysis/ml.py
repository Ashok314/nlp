import fasttext
model = fasttext.train_supervised(input="train", lr=1.0, epoch=1000,wordNgrams=2,bucket=200000,dim=100, loss='ova')
testparam=model.test("test",k=10)
model.save_model("model.bin")


print("validation params",testparam)
#print(model.predict(""))

print(model.predict("越後米つき"))