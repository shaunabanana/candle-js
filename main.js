import {Model, Loader, Tensor, Linear, Conv2d, ConvTranspose2d, ReLU, Sigmoid} from "./candle.js";


class FontVAE extends Model {

    constructor () {
        super();
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

    encode(image) {
        let h = ReLU(this.dconv1.forward(image));
        h = ReLU(this.dconv2.forward(h));
        h = ReLU(this.dconv3.forward(h));
        h = ReLU(this.dconv4.forward(h));
        h = h.flatten().unsqueeze(0);
        return {
            mu: this.proj_mu.forward(h),
            logvar: this.proj_logvar.forward(h)
        }
    }

    decode(z) {
        let h = this.proj_z.forward(z).transpose(0, 1).unsqueeze(2);
        h = ReLU(this.uconv1.forward(h));
        h = ReLU(this.uconv2.forward(h));
        h = ReLU(this.uconv3.forward(h));
        return Sigmoid(this.uconv4.forward(h));
    }
}
let vae = new FontVAE();


let loader = new Loader('model-candle');
loader.load(vae, function () {
    console.log('Everything loaded');
    let image = new Tensor(1, 64, 64).fill(1);
    let z = new Tensor(1, 126).fill(0);
    z.set(1, 0, 64);
    console.log(z);
    console.log(vae.encode(image));
    let recon = vae.decode(z);
    console.log(recon);
    console.log(recon.get(0, 0, 0));
    console.log(recon.get(0, 0, 5));
    console.log(recon.get(0, 1, 5));

});


new p5(( p ) => {

    let x = 100;
    let y = 100;
  
    p.setup = () => {
        p.createCanvas(200, 200);
    };
  
    p.draw = () => {
        p.background(0);
        p.fill(255);
        p.rect(x,y,50,50);
    };
});