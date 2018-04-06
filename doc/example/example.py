import pconsc4

model = pconsc4.get_model()

pred_1 = pconsc4.predict(model, 'alignment1')
pred_2 = pconsc4.predict(model, 'alignment2')