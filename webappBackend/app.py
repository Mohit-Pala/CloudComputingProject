from flask import Flask, request, jsonify
import os
import torch
import base64
from io import BytesIO
from torchvision import transforms
import torch.nn as nn
import torchvision.utils as vutils

workers = 2
batch_size = 128
image_size = 64
nc = 3
nz = 100
ngf = 64
ndf = 64
num_epochs = 50
lr = 0.0002
beta1 = 0.5
ngpu = 0


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)


device = torch.device("cuda:0" if (
    torch.cuda.is_available() and ngpu > 0) else "cpu")
netG = Generator(ngpu).to(device)


app = Flask(__name__)


@app.route('/', methods=['GET'])
def helloWorld():
    netG = Generator(ngpu).to(device)
    netG.load_state_dict(torch.load('generator_model.pth', map_location=torch.device('cpu')))
    netG.eval()  # Set the model to evaluation mode

    # Generate a random noise vector
    noise = torch.randn(1, nz, 1, 1, device=device)

    # Generate the image using the loaded generator
    with torch.no_grad():
        generated_image = netG(noise).detach().cpu()
        
    # Save the generated image to a file
    # Convert the image to base64 string
    
    # Normalize the image to [0,1] range
    img = vutils.make_grid(generated_image, normalize=True)
    
    # Convert to PIL image
    img_pil = transforms.ToPILImage()(img)
    
    # Save to BytesIO object
    buffer = BytesIO()
    img_pil.save(buffer, format="PNG")
    
    # Convert to base64 string
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # Resize to 128x128
    img_resized = transforms.Resize((128, 128))(img_pil)
    
    # Save to BytesIO object
    buffer = BytesIO()
    img_resized.save(buffer, format="PNG")
    
    # Convert to base64 string
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # Return base64 string in an HTML img tag
    return f'<img src="data:image/png;base64,{img_str}" />'

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
