let gpu;
if (navigator.platform == 'Win32') {
    gpu = new GPU({mode: 'dev'});
} else {
  gpu = new GPU();
}

let config = {
    dynamicOutput: true, 
    dynamicArguments: true,
    tactic: 'precision',
    fixIntegerDivisionAccuracy: true,
    loopMaxIterations: 100000000
}

const addNumberElementwize = gpu.createKernel(function(mat, num) {
    return mat[this.thread.x] + num;
}, config);

const subtractNumberElementwize = gpu.createKernel(function(mat, num) {
    return mat[this.thread.x] - num;
}, config);

const multiplyNumberElementwize = gpu.createKernel(function(mat, num) {
    return mat[this.thread.x] * num;
}, config);

const divideNumberElementwize = gpu.createKernel(function(mat, num) {
    return mat[this.thread.x] / num;
}, config);

const powerNumberElementwize = gpu.createKernel(function(mat, num) {
    return Math.pow(mat[this.thread.x], num);
}, config);

const addTensorElementwize = gpu.createKernel(function(mat1, mat2) {
    return mat1[this.thread.x] + mat2[this.thread.x];
}, config);

const addTensorBroadcast2d = gpu.createKernel(function(mat1, mat2, dim) {
    if (dim == 0) {
        return mat1[this.thread.y][this.thread.x] + mat2[0][this.thread.x];
    } else if (dim == 1) {
        return mat1[this.thread.y][this.thread.x] + mat2[this.thread.y][0];
    }
    return 0;
}, config);

const addTensorBroadcast3D = gpu.createKernel(function(mat1, mat2, dim) {
    console.log(this.thread.x, this.thread.y, this.thread.z);
    /*
    if (dim == 0) {
        return mat1[this.thread.z][this.thread.y][this.thread.x] + mat2[this.thread.z][this.thread.y][0];
    } else if (dim == 1) {
        return mat1[this.thread.z][this.thread.y][this.thread.x] + mat2[this.thread.z][0][this.thread.x];
    } else if (dim == 2) {
        return mat1[this.thread.z][this.thread.y][this.thread.x] + mat2[0][this.thread.y][this.thread.x];
    }
    */
    return 0;
}, config);

const subtractTensorElementwize = gpu.createKernel(function(mat1, mat2) {
    return mat1[this.thread.x] - mat2[this.thread.x];
}, config);

const multiplyTensorElementwize = gpu.createKernel(function(mat1, mat2) {
    return mat1[this.thread.x] * mat2[this.thread.x];
}, config);

const divideTensorElementwize = gpu.createKernel(function(mat1, mat2) {
    return mat1[this.thread.x] / mat2[this.thread.x];
}, config);

const multiplyTensorMatrix = gpu.createKernel(function(mat1, mat2, length) {
    let sum = 0;
    for (let i = 0; i < length; i++) {
        sum += mat1[this.thread.y][i] * mat2[i][this.thread.x];
    }
    return sum;
}, config);

const transposeTensorMatrix2d = gpu.createKernel(function(mat, dim0, dim1) {
    return mat[this.thread.x][this.thread.y];
}, config);

const transposeTensorMatrix3D = gpu.createKernel(function(mat, dim0, dim1) {
    if (dim0 == 0 && dim1 == 1 || dim0 == 1 && dim1 == 0) {
        return mat[this.thread.y][this.thread.z][this.thread.x];
    } else if (dim0 == 0 && dim1 == 2 || dim0 == 2 && dim1 == 0) {
        return mat[this.thread.x][this.thread.y][this.thread.z];
    } else if (dim0 == 1 && dim1 == 2 || dim0 == 2 && dim1 == 1) {
        return mat[this.thread.z][this.thread.x][this.thread.y];
    }
}, config);

const applyReLU = gpu.createKernel(function(mat) {
    if (mat[this.thread.x] > 0) {
        return mat[this.thread.x];
    } else {
        return 0;
    }
}, config);

const applySigmoid = gpu.createKernel(function(mat) {
    return 1 / (1 + Math.exp(-mat[this.thread.x]));
}, config);


function arrayEqual(a, b) {
    if (a === b) return true;
    if (a == null || b == null) return false;
    if (a.length != b.length) return false;
    for (var i = 0; i < a.length; ++i) {
      if (a[i] !== b[i]) return false;
    }
    return true;
}


export class Tensor {
    constructor (...dims) {
        this.D = dims.slice();  // Dimension
        this.S = [];    // Stride
        if (dims.length > 0) {
            dims.reverse()
            let size = 1;
            for (let dim of dims) {
                this.S.push(size);
                size *= dim;
            }
            this.S = this.S.reverse();
            this.T = new Array(size);
        }
    }

    copy() {
        let t = new Tensor(...this.D);
        t.T = this.T.slice();
        t.G = this.G.slice();
        return t;
    }

    size (dim) {
        if (typeof dim === 'undefined') {
            return this.D;
        } else if (typeof dim === 'number' && dim >= this.D.length) {
            throw `Tensor of size [${this.D}] does not have a dimension ${dim}.`;
        } else if (typeof dim === 'number') {
            return this.D[dim];
        } else {
            throw `Incorrect dimension ${dim}.`;
        }
    }

    fill (input) {
        if (typeof input === 'number') {
            this.T.fill(input);
        } else if (typeof input === 'function') {
            this.T = [...this.T].map(input);
        }
        return this;
    }

    transpose (dim1, dim2) {
        let dims = this.D.slice();
        let transposeFunction;
        if (dims.length > 3) {
            throw `Tranpose of tensor with more than three dimensions are not supported yet.`
        }
        let outputDims = dims.slice();
        let tmpDim = outputDims[dim1];
        outputDims[dim1] = outputDims[dim2];
        outputDims[dim2] = tmpDim;

        if (dims.length == 2) {
            transposeFunction = transposeTensorMatrix2d;
        } else if (dims.length == 3) {
            transposeFunction = transposeTensorMatrix3D;
        }
        transposeFunction.setOutput(outputDims.reverse());
        let out = transposeFunction(
            GPU.input(this.T, dims.reverse()),
            dim1, dim2
        )
        let result = new Float32Array(this.T.length);
        GPU.utils.flattenTo(out, result);
        let t = new Tensor(...dims);
        t.T = [...result];
        return t;
    }

    unsqueeze(dim) {
        let dims = this.D.slice(0, dim).concat([1]).concat(this.D.slice(dim, this.D.length));
        let t = new Tensor(...dims);
        t.T = this.T.slice();
        return t;
    }

    flatten() {
        let t = new Tensor(this.T.length);
        t.T = this.T.slice();
        return t;
    }

    get (...indices) {
        let pos = 0;
        for (let i in indices) {
            if (indices[i] >= this.D[i]) {
                throw `Index ${indices[i]} at dimension ${i} is too large for tensor of size [${this.D}].`
            }
            pos += this.S[i] * indices[i];
        }
        return this.T[pos];
    }

    set (value, ...indices) {
        let pos = 0;
        for (let i in indices) {
            if (indices[i] >= this.D[i]) {
                throw `Index ${indices[i]} at dimension ${i} is too large for tensor of size [${this.D}].`
            }
            pos += this.S[i] * indices[i];
        }
        this.T[pos] = value;
    }

    static fromArray(array) {
        let current = array;
        let dims = [];
        while (Array.isArray(current)) {
            dims.push(current.length);
            current = current[0];
        }
        while (Array.isArray(array[0])) {
            array = array.flat();
        }
        let t = new Tensor(...dims);
        t.T = array;
        return t;
    }

    static ones (...dims) {
        return new Tensor(...dims).fill(1);
    }

    static zeros (...dims) {
        return new Tensor(...dims).fill(0);
    }

    /*
    static broadcastable (tensor1, tensor2) {
        let ind1 = tensor1.D.length - 1;
        let ind2 = tensor2.D.length - 1;
        while (ind1 >= 0 && ind2 >= 0) {
            if (tensor1.D[ind1] > 1 && tensor2.D[ind2] > 1 && tensor1.D[ind1] != tensor2.D[ind2]) {
                return false;
            }
            ind1 --;
            ind2 --;
        }
        return true;
    }
    */

    add (other, dim) {
        if (typeof other === 'number') {
            addNumberElementwize.setOutput([this.T.length]);
            let t = new Tensor(...this.D);
            t.T = [...addNumberElementwize(this.T, other)];
            return t;
        } else if (other instanceof Tensor) {
            if (arrayEqual(this.D, other.D)) {
                addTensorElementwize.setOutput([this.T.length]);
                let t = new Tensor(...this.D);
                t.T = [...addTensorElementwize(this.T, other.T)];
                return t;
            } else if (typeof dim === 'number') {
                let dim1 = this.D.slice(0, dim).concat(this.D.slice(dim + 1, this.D.length));
                let dim2 = other.D.slice(0, dim).concat(other.D.slice(dim + 1, this.D.length));
                if (!arrayEqual(dim1, dim2)) {
                    throw `Expected two tensors to have equal dimensions except at dimension ${dim}. Got [${this.D}] and [${other.D}]`
                } else if (other.D[dim] != 1) {
                    throw `Expected the other tensor to have 1 at dimension ${dim}. Got ${other.D[dim]}.`
                }
                addTensorBroadcast2d.setOutput(this.D.slice().reverse());
                let out = addTensorBroadcast2d(
                    GPU.input(this.T, this.D.slice().reverse()),
                    GPU.input(other.T, other.D.slice().reverse()),
                    dim
                )
                let result = new Float32Array(this.T.length);
                GPU.utils.flattenTo(out, result);
                let t = new Tensor(...this.D);
                t.T = [...result];
                return t;
            } else {
                throw `Tensor size mismatch: [${this.D}] and [${other.D}].`;
            }
        } else {
            throw `Wrong type in add(). Got ${other}.`;
        }
    }

    sub (other) {
        if (typeof other === 'number') {
            subtractNumberElementwize.setOutput([this.T.length]);
            let t = new Tensor(...this.D);
            t.T = [...subtractNumberElementwize(this.T, other)];
            return t;
        } else if (other instanceof Tensor) {
            if (!arrayEqual(this.D, other.D)) {
                throw `Tensor size mismatch: [${this.D}] and [${other.D}].`;
            }
            subtractTensorElementwize.setOutput([this.T.length]);
            let t = new Tensor(...this.D);
            t.T = [...subtractTensorElementwize(this.T, other.T)];
            return t;
        }
    }

    mul (other) {
        if (typeof other === 'number') {
            multiplyNumberElementwize.setOutput([this.T.length]);
            let t = new Tensor(...this.D);
            t.T = [...multiplyNumberElementwize(this.T, other)];
            return t;
        } else if (other instanceof Tensor) {
            if (!arrayEqual(this.D, other.D)) {
                throw `Tensor size mismatch: [${this.D}] and [${other.D}].`;
            }
            multiplyTensorElementwize.setOutput([this.T.length]);
            let t = new Tensor(...this.D);
            t.T = [...multiplyTensorElementwize(this.T, other.T)];
            return t;
        }
    }

    div (other) {
        if (typeof other === 'number') {
            divideNumberElementwize.setOutput([this.T.length]);
            let t = new Tensor(...this.D);
            t.T = [...divideNumberElementwize(this.T, other)];
            return t;
        } else if (other instanceof Tensor) {
            if (!arrayEqual(this.D, other.D)) {
                throw `Tensor size mismatch: [${this.D}] and [${other.D}].`;
            }
            divideTensorElementwize.setOutput([this.T.length]);
            let t = new Tensor(...this.D);
            t.T = [...divideTensorElementwize(this.T, other.T)];
            return t;
        }
    }

    pow (exp, grad) {
        if (typeof exp === 'number') {
            powerNumberElementwize.setOutput([this.T.length]);
            let t = new Tensor(...this.D);
            t.T = [...powerNumberElementwize(this.T, exp)];
            if (typeof grad === 'boolean' && !grad) return t;
            return t;
        } else {
            throw `${exp} is not a number!`;
        }
    }

    matmul (other) {
        if ( !(other instanceof Tensor) ) {
            throw 'The object to matmul is not a Tensor.';
        } else if (this.D.length > 3 || other.D.length > 3) {
            throw `Can only matmul Tensors of size up to [*, *]. Has ${this.D.length} and ${other.D.length}.`;
        }

        if (this.D.length == 1 && other.D.length == 1) {
            return this.mul(other).sum();
        } else if (this.D.length == 2 && other.D.length == 2) {
            if (this.D[1] != other.D[0]) {
                throw `Columns of array 1 and rows of array 2 do not match! Got ${this.D[1]} and ${other.D[0]}`;
            }
            multiplyTensorMatrix.setOutput([other.D[1], this.D[0]]);
            let out = multiplyTensorMatrix(
                GPU.input(this.T, [this.D[1], this.D[0]]),
                GPU.input(other.T, [other.D[1], other.D[0]]),
                this.D[1]
            )
            let result = new Float32Array(this.D[0] * other.D[1]);
            GPU.utils.flattenTo(out, result);
            let t = new Tensor(this.D[0], other.D[1]);
            t.T = [...result];
            return t;
        } else {
            console.log(this, other);
            throw `Not implemented.`;
        }
    }

    sum () {
        let t = new Tensor(1);
        t.T = [this.T.reduce((pv, cv) => pv + cv, 0)];
        return t;
    }

    mean () {
        let t = new Tensor(1);
        t.T = [this.T.reduce((pv, cv) => pv + cv, 0) / this.T.length];
        return t;
    }

    std () {
        let t = new Tensor(1);
        t.T = [Math.sqrt(this.sub(this.mean()).pow(2).sum() / (this.T.length - 1))];
        return t;
    }

    get nElements () {
        return this.T.length;
    }
}

export class Model {
    constructor () {}
    forward () {}
    backward () {}
}

export class Linear extends Model {
    constructor (dimIn, dimOut) {
        super();
        this.dimIn = dimIn;
        this.dimOut = dimOut;

        this.weight = new Tensor(dimOut, dimIn).fill(() => randomGaussian());
        this.bias = new Tensor(1, dimOut).fill(() => randomGaussian());
    }

    forward (input) {
        return input.matmul(this.weight.transpose(0, 1)).add(this.bias, 0);
    }
}

const padTensorMatrix2d = gpu.createKernel(function(mat, width, height, padding) {
    if (this.thread.x < padding || this.thread.x - width > 0) {
        return 0;
    }
    if (this.thread.y < padding || this.thread.y - height > 0) {
        return 0;
    }
    return mat[this.thread.z][this.thread.y - padding][this.thread.x - padding];
}, config);

const convoluteTensorMatrix2d = gpu.createKernel(function(mat, conv, bias, kernelSize, stride, inChannels) {
    let center = Math.ceil(kernelSize / 2);
    let x = this.thread.x * stride + center;
    let y = this.thread.y * stride + center;

    let sum = 0;
    for (var i = 0; i < inChannels; i++) {
        for (var c = 0; c < kernelSize * kernelSize; c++) {
            let yoff = Math.floor(c / kernelSize) - center;
            let xoff = c % kernelSize - center;
            sum += mat[i][y + yoff][x + xoff] * conv[this.thread.z][i][c];
        }
    }
    return sum + bias[this.thread.z];
}, config);

export class Conv2d extends Model {
    constructor (inChannels, outChannels, kernelSize, stride=1, padding=0, dilation=1) {
        super();
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;
        this.dilation = dilation;
        if (padding > 0) throw 'Dilation not yet supported in ConvTranspose2d.'

        this.weight = new Tensor(outChannels, inChannels, kernelSize, kernelSize).fill(() => randomGaussian());
        this.bias = new Tensor(outChannels).fill(() => randomGaussian());
    }

    forward (input) {
        padTensorMatrix2d.setOutput([input.D[2] + this.padding * 2, input.D[1] + this.padding * 2, input.D[0]]);
        let padded = padTensorMatrix2d(
            GPU.input(input.T, input.D.slice().reverse()),
            input.D[2],
            input.D[1],
            this.padding
        );

        let outWidth = Math.floor( (input.D[2] + 2 * this.padding - this.dilation * (this.kernelSize - 1) - 1) / this.stride + 1 )
        let outHeight = Math.floor( (input.D[1] + 2 * this.padding - this.dilation * (this.kernelSize - 1) - 1) / this.stride + 1 )
        let outputDims = [this.outChannels, outHeight, outWidth];

        convoluteTensorMatrix2d.setOutput(outputDims.slice().reverse());
        
        let out = convoluteTensorMatrix2d(
            padded, 
            GPU.input(this.weight.T, [this.weight.D[3] * this.weight.D[2], this.weight.D[1], this.weight.D[0]]),
            this.bias.T,
            this.kernelSize, this.stride, this.inChannels
        );

        let result = new Float32Array(this.outChannels * outHeight * outWidth);
        GPU.utils.flattenTo(out, result);
        let t = new Tensor(...outputDims);
        t.T = [...result];
        return t;
    }
}

const transposeConvoluteTensorMatrix2d = gpu.createKernel(function(mat, conv, bias, inWidth, inHeight, kernelSize, stride, inChannels) {
    let sum = 0;
    for (var h = 0; h < inHeight; h++) {
        for (var w = 0; w < inWidth; w++) {
            let startX = w * stride;
            let endX = startX + kernelSize;
            let startY = h * stride;
            let endY = startY + kernelSize;
            if (this.thread.x >= startX && this.thread.x < endX && this.thread.y >= startY && this.thread.y < endY) {
                let kernelX = this.thread.x - startX;
                let kernelY = this.thread.y - startY;
                let kernelPos = kernelY * kernelSize + kernelX;
                for (var i = 0; i < inChannels; i++) {
                    sum += mat[i][h][w] * conv[i][this.thread.z][kernelPos];
                }
            }
        }
    }
    return sum + bias[this.thread.z];
}, config);

export class ConvTranspose2d extends Model {
    constructor (inChannels, outChannels, kernelSize, stride=1, padding=0, dilation=1) {
        super();
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;
        if (padding > 0) throw 'Padding not yet supported in ConvTranspose2d.'
        this.dilation = dilation;
        if (padding > 0) throw 'Dilation not yet supported in ConvTranspose2d.'

        this.weight = new Tensor(inChannels, outChannels, kernelSize, kernelSize).fill(1);
        this.bias = new Tensor(outChannels).fill(1);
    }

    forward (input) {
        let outWidth = (input.D[2] - 1) * this.stride - 2 * this.padding + this.dilation * (this.kernelSize - 1) + 0 + 1;
        let outHeight = (input.D[1] - 1) * this.stride - 2 * this.padding + this.dilation * (this.kernelSize - 1) + 0 + 1;
        let outputDims = [this.outChannels, outHeight, outWidth];

        transposeConvoluteTensorMatrix2d.setOutput(outputDims.slice().reverse());
        let out = transposeConvoluteTensorMatrix2d(
            GPU.input(input.T, input.D.slice().reverse()),
            GPU.input(this.weight.T, [this.weight.D[3] * this.weight.D[2], this.weight.D[1], this.weight.D[0]]),
            this.bias.T,
            input.D[2], input.D[1],
            this.kernelSize, this.stride, this.inChannels
        );

        let result = new Float32Array(this.outChannels * outHeight * outWidth);
        GPU.utils.flattenTo(out, result);
        let t = new Tensor(...outputDims);
        t.T = [...result];
        return t;
    }
}

export function ReLU(input) {
    applyReLU.setOutput([input.T.length]);
    let t = new Tensor(...input.D);
    t.T = [...applyReLU(input.T)]
    return t;
}

export function Sigmoid(input) {
    applySigmoid.setOutput([input.T.length]);
    let t = new Tensor(...input.D);
    t.T = [...applySigmoid(input.T)]
    return t;
}

const TWO_PI = Math.PI * 2;
export function randomGaussian(mean=0, std=1){
    var u1 = Math.random();
    var u2 = Math.random();
    
    var z = Math.sqrt(-2.0 * Math.log(u1)) * Math.sin(TWO_PI * u2);

    return z * std + mean;
}















let joinPath = (...args) => {
    return args.map((part, i) => {
        if (i === 0) {
            return part.trim().replace(/[\/]*$/g, '')
        } else {
            return part.trim().replace(/(^[\/]*|[\/]*$)/g, '')
        }
    }).filter(x=>x.length).join('/')
}

export class Loader {

    constructor(address, requests=50) {
        this.address = address;
        this.requests = requests;
        this.chunks = {};
        this.data = [];

        this.loaded = 0;
        this.finished = false;
        this.percent = 0;
        
        // this.total = this.files.length;
        this.totalLoaded = 0;
    }

    loadMetadata(callback) {
        let req = new XMLHttpRequest();
        req.open("GET", joinPath(this.address, 'model.json'), true);

        req.onload = function () {
            if (callback) callback(JSON.parse(req.responseText));
        };
        req.send(null);
    }

    loadOne(url, callback, errorCallback) {
        let oReq = new XMLHttpRequest();
        oReq.open("GET", url, true);
        oReq.responseType = "arraybuffer";

        oReq.onload = function () {
            var arrayBuffer = oReq.response; // Note: not oReq.responseText
            if (arrayBuffer) {
                var byteArray = new Uint8Array(arrayBuffer);
                if (callback) callback(Loader.floatArray(byteArray));
            }
        };
        oReq.onerror = function () {
            if (errorCallback) errorCallback(url);
        };
        oReq.send(null);
    }

    _processChunks() {
        this.files = [...Array(this.metadata.chunks).keys()].map((i) => i + '.bin');
        this.files.forEach((name) => {
            let chunk = this.chunks[name];
            let len = chunk.length;
            for (let i = 0; i < len; i++ ) {
                this.data.push(chunk[i]);
            }
        });

        let layers = Object.keys(this.metadata.layers);
        for (let layer of layers) {
            let params = Object.keys(this.metadata.layers[layer]);
            for (let param of params) {
                this.model[layer][param].T = this.data.slice(
                    this.metadata.layers[layer][param].start,
                    this.metadata.layers[layer][param].end
                )
            }
        }
    }

    _load(callback) {
        this.loaded = 0;
        this.currentFiles = this.files.splice(0, this.requests);
        this.currentFiles.forEach(function (url) {
            this.loadOne(
                joinPath(this.address, url), 

                function (data) {
                    this.chunks[url] = data;
                    this.loaded++;
                    this.totalLoaded++;
                    this.percent = this.totalLoaded / this.metadata.chunks;
                    if (this.loaded >= this.currentFiles.length) {
                        if (this.files.length <= 0) {
                            this.finished = true;
                            this._processChunks();
                            if (callback) callback();
                        } else {
                            this._load(callback);
                        }
                    }
                }.bind(this), 

                function (url) {
                    print('error loading ' + url + ', restarting')
                    this.files.push(url);
                }.bind(this)
            );
        }.bind(this));
    }

    load(model, callback) {
        this.model = model;
        this.loadMetadata(function (metadata) {
            this.metadata = metadata;
            this.files = [...Array(metadata.chunks).keys()].map((i) => i + '.bin');
            this._load(callback);
        }.bind(this));
    }

    static floatArray(bytes) {
        let arr = [];
        for (var i = 0; i < bytes.length; i += 4) {
            arr.push(this.toFloat(bytes.slice(i, i + 4), false))
        }
        return arr;
    }

    static toFloat(bytes) {
        // Reference: https://stackoverflow.com/questions/42699162/javascript-convert-array-of-4-bytes-into-a-float-value-from-modbustcp-read
        var buf = new ArrayBuffer(4);
        var view = new DataView(buf);
        bytes.reverse().forEach(function (b, i) {
            view.setUint8(i, b);
        });
        return view.getFloat32(0);
    }

}