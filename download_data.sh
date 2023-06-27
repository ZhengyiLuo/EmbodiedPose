mkdir sample_data
mkdir sample_data/humor
mkdir sample_data/res_wild/
gdown https://drive.google.com/uc?id=12rxODUmyOyOg11rQUY7GTmT-NDH7R2uN -O  sample_data/ # standing netural
gdown https://drive.google.com/uc?id=1tfZEob1wxIwWj2fcLiml0Dj9O89eP20w -O  sample_data/ # Processed PROX camera & 2D keypoint detecitons
gdown https://drive.google.com/uc?id=1axNSXL2uGuEbFnYA2pECZy8PN5bk7VaR -O  sample_data/ # kinpoly data
gdown https://drive.google.com/uc?id=1BvRl4PaLOf7l4tGzu2I2-sM6J0ER1MQQ -O  sample_data/ # h36m fitted
gdown https://drive.google.com/uc?id=1GdRmh_J6cC0SDJJLqrl6aAQtfNNqWnr9 -O  sample_data/res_wild/res.pkl # wild fitted
gdown https://drive.google.com/uc?id=1i6fr2qlYiEvZb8pTudAzH6zkbuunrvSk -O  sample_data/humor/ # pretrained humor model
gdown https://drive.google.com/uc?id=16Nws1EmlMm35T-GlXpapJ7EAv8SbCG3R -O  sample_data/prox/qualitative/ && unzip sample_data/prox/qualitative/prox_calibration.zip -d sample_data/prox/qualitative/ 
gdown https://drive.google.com/uc?id=1CYX0_4NLznaHW7JFpYQ9zpxXTyc5ylLe -O  results/scene+/tcn_voxel_4_5/models/ # pretrained EmbodiedPose policy


