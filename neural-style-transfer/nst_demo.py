import pystiche
from pystiche import enc,loss,ops,optim,demo
from pystiche.image import write_image,read_image
from pystiche.misc import get_device, get_input_image

print("I'm working with pystiche=={}".format(pystiche.__version__))
device = get_device()
print("I'm working with device=={}".format(device))

multi_layer_encoder = enc.vgg19_multi_layer_encoder()
print(multi_layer_encoder)

criterion = loss.PerceptualLoss(content_loss=ops.FeatureReconstructionOperator(multi_layer_encoder.extract_encoder("relu4_2")),
    style_loss=ops.MultiLayerEncodingOperator(multi_layer_encoder,("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"),
    lambda encoder, layer_weight: ops.GramOperator(encoder, score_weight=layer_weight),score_weight=1e3),).to(device)

size = 500

content_image = read_image('demo_data/content.png',device=device,size=size)
style_image = read_image('demo_data/style.png',device=device,size=size)

criterion.set_content_image(content_image)
criterion.set_style_image(style_image)

starting_point = "content"
input_image = get_input_image(starting_point, content_image=content_image)

output_image = optim.image_optimization(input_image, criterion, num_steps=500)

write_image(output_image,'demo_data/result.png')
