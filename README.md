# Candle.js
While there's no Torch in the browser, there's still Candle.
Candle.js is a GPU-accelerated library to make running PyTorch models in the browser a bit easier. It also serves as a tensor computation library.

## Usage example

### Import dependencies
Import gpu.js as Candle.js depends on it for accelerated computing:

```html
<head>
    ...
    <script src='libs/gpu-browser.min.js'></script>
</head>
```

Import classes from Candle.js:

```javascript
import {Model, Loader, Tensor, Linear, Conv2d, ConvTranspose2d, ReLU, Sigmoid} from "./candle.js";
```

### Model definition

> **TL;DR: The layer names should match!**

**Your PyTorch model:**

```python
def __init__(...):
    ...
    self.dconv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2)
    self.dconv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
    self.dconv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
    self.dconv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2)
    self.proj_mu = nn.Linear(1024, 64)
    self.proj_logvar = nn.Linear(1024, 64)

    self.proj_z = nn.Linear(126, 1024)
    self.uconv1 = nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2)
    self.uconv2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2)
    self.uconv3 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2)
    self.uconv4 = nn.ConvTranspose2d(32, 1, kernel_size=6, stride=2)
```

**Your Candle.js model:**

```javascript
constructor (...) {
    ...
    this.dconv1 = new Conv2d(1, 32, 4, 2);
    this.dconv2 = new Conv2d(32, 64, 4, 2);
    this.dconv3 = new Conv2d(64, 128, 4, 2);
    this.dconv4 = new Conv2d(128, 256, 4, 2);
    this.proj_mu = new Linear(1024, 64);
    this.proj_logvar = new Linear(1024, 64);

    this.proj_z = new Linear(126, 1024);
    this.uconv1 = new ConvTranspose2d(1024, 128, 5, 2);
    this.uconv2 = new ConvTranspose2d(128, 64, 5, 2);
    this.uconv3 = new ConvTranspose2d(64, 32, 6, 2);
    this.uconv4 = new ConvTranspose2d(32, 1, 6, 2);
}
```

### Forward function
**Your PyTorch model:**

```python
def forward(x):
    h = F.relu(self.dconv1(x))
    ...
```

**Your Candle.js model:**

```javascript
forward (x) {
    let h = ReLU(this.dconv1.forward(x));
    ...
}
```

### Loading a pretrained model generated with Candle.js helper
**Using Candle.js helper:**

```
python candle-helper.py /path/to/your/model.pt
```

This will create a directory named `/path/to/your/model-candle`. In the directory will be a model.json that describes you model, and chunks of binary data that will be loaded by the Candle.js Loader accordingly.

**Then, in your script:**

```javascript
let loader = new Loader('url/to/model-candle');
loader.load(vae, function () {
    console.log('Everything loaded!');
});
```

Et voila! You can now run your PyTorch model in the browser!

## Currently supported layers (more layers to come!):

> **Note: currently no batch processing is supported in convolutional layers due to limits in GPU acceleration.** 
>
> That is, Conv2d and ConvTranspose2d cannot accept inputs with size (n_samples, channels, height, width). Instead, pass in tensors one by one, with size (channels, height, width).

* Linear
* Conv2d (no dilation support)
* ConvTranspose2d (no padding and dilation support)
* ReLU
* Sigmoid
