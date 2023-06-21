# INPUT

## INPUT_model

`name` : model name,

`model_path`: Import existing model, Input model path;"False",

`epoch`: Model training rounds,

`batch_size`: batch size,

`lr`: learning rate,

`print_step`:How many steps to print the loss function after training,

`save_step`:How many steps to save model after training,

`seed`: random seed,

`dir`: "./",

`loss_type`: "Zernikeloss","intensityloss","Zernike+intensityloss",

`save`: Whether to store training data
---
--- 
## INPUT_propagation

`mm`: $mm = log_{2}(n_{grid}))$.

`mgs` : Truncated beams. pow( , mgs) = pow( ,truncated beams). 

`a0` :  The spot radius of initial fields.

`xx0` : Multiple of initial light field buffer area. 

`plm` :  Wave length.

`zfh` : Transmission distance.

`xxz` : Multiple of focal light field buffer area.

`minZnkDim` : Minimum order of a polynomial.

`maxZnkOrder` : Maximum degree of polynomial(MAX 13).

`rms` : Phase variance.

`eeznk` :  Polynomial coefficient variance change index. 

`zernike_dir`: "random","zernike_dir"

`nsnapshot`:frame number
---
---
