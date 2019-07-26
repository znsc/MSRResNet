## [AIM Workshop and Challenge](http://www.vision.ee.ethz.ch/aim19/) @ ICCV 2019
## Constrained Super-Resolution Challenge

Jointly with AIM workshop we have an AIM challenge on Constrained Super-Resolution, that is, the task of super-resolving (increasing the resolution) an input image with a magnification factor x4 based on a set of prior examples of low and corresponding high resolution images. The challenge has three tracks.

**[Track 1: Parameters](https://competitions.codalab.org/competitions/20167)**, the aim is to obtain a network design / solution with the lowest amount of parameters while being constrained to maintain or improve the PSNR result and the inference time (runtime) of MSRResNet (Ledig et al, 2017 & Wang et al, 2018).

**[Track 2: Inference](https://competitions.codalab.org/competitions/20168)**, the aim is to obtain a network design / solution with the lowest inference time (runtime) on a common GPU (ie. Titan Xp) while being constrained to maintain or improve over MSRResNet (Ledig et al, 2017 & Wang et al, 2018) in terms of number of parameters and the PSNR result.

**[Track 3: Fidelity](https://competitions.codalab.org/competitions/20169)**, the aim is to obtain a network design / solution with the best fidelity (PSNR) while being constrained to maintain or improve over MSRResNet (Ledig et al, 2017 & Wang et al, 2018) in terms of number of parameters and inference time on a common GPU (ie. Titan Xp).

## Baseline model (MSRResNet)

* Number of parameters: 1,517,571 (1.5M)

    ```
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    ```

* Average PSNR on validation data: 29.00 dB

* Average inference time (Titan Xp) on validation data: 0.170 second 

    Note: I selected the best average inference time among three trials

Run [test_demo.py](test_demo.py) to test the model
