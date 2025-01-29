from typing import Literal, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def LSregress(
    pred: torch.Tensor, gt: torch.Tensor, origin: torch.Tensor
) -> torch.Tensor:
    """
    Returns
    -------
    v
        A slightly more complex version of ``sum_along_all_dims_except_batch(pred)/sum_along_all_dims_except_batch(gt)*origin``
    """
    nb = pred.size(0)
    origSize = pred.size()
    pred = pred.reshape(nb, -1)
    gt = gt.reshape(nb, -1)

    coef = (
        torch.sum(pred * gt, dim=1)
        / torch.clamp(torch.sum(pred * pred, dim=1), min=1e-5)
    ).detach()
    coef = torch.clamp(coef, 0.001, 1000)
    for n in range(0, len(origSize) - 1):
        coef = coef.unsqueeze(-1)
    pred = pred.view(origSize)

    predNew = origin * coef.expand(origSize)

    return predNew


def LSregressDiffSpec(
    diff: torch.Tensor,
    spec: torch.Tensor,
    imOrig: torch.Tensor,
    diffOrig: torch.Tensor,
    specOrig: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    A function similar to `LSregress`, but implemented for the diffuse and specular maps and it also properly weights them.

    """
    # batch size, number of channels, image height, image width
    nb, nc, nh, nw = diff.size()

    # Mask out too bright regions
    mask = (imOrig < 0.9).float()
    diff = diff * mask
    spec = spec * mask
    im = imOrig * mask

    diff = diff.reshape(nb, -1)
    spec = spec.reshape(nb, -1)
    im = im.reshape(nb, -1)

    R"""
    Calculate the weights of the diffuse and the specular component to approximate the original image as good as possible.

    Equation:

    [ sum(diff*diff)     sum(diff*spec) ] x = [ sum(diff*im) ]
    [ sum(spec*diff)     sum(spec*spec) ]     [ sum(spec*im) ]
    \---------------  ------------------/     \------  ------/
                    \/                               \/
                    A                                b

    Slightly reworded version:

    [ mean(diff*diff)     mean(diff*spec) ] x = [ mean(diff*im) ]
    [ mean(spec*diff)     mean(spec*spec) ]     [ mean(spec*im) ]

    Intuition: the sum of the diffuse and specular component should be equal to the original image.

    Behavior of the determinant of A:

    * if the variance of the maps is low --> the determinant is low
    * if the variance of the maps is high --> the determinant is high
    """
    a11 = torch.sum(diff * diff, dim=1)
    a22 = torch.sum(spec * spec, dim=1)
    a12 = torch.sum(diff * spec, dim=1)

    frac = a11 * a22 - a12 * a12  # determinant of A
    b1 = torch.sum(diff * im, dim=1)
    b2 = torch.sum(spec * im, dim=1)

    # Compute the coefficients based on linear regression
    coef1 = b1 * a22 - b2 * a12
    coef2 = -b1 * a12 + a11 * b2
    coef1 = coef1 / torch.clamp(frac, min=1e-2)
    coef2 = coef2 / torch.clamp(frac, min=1e-2)

    # Compute the coefficients assuming diffuse albedo only
    coef3 = torch.clamp(b1 / torch.clamp(a11, min=1e-5), 0.001, 1000)
    coef4 = coef3.clone() * 0

    frac = (frac / (nc * nh * nw)).detach()
    fracInd = (frac > 1e-2).float()

    coefDiffuse = fracInd * coef1 + (1 - fracInd) * coef3
    coefSpecular = fracInd * coef2 + (1 - fracInd) * coef4

    for n in range(0, 3):
        coefDiffuse = coefDiffuse.unsqueeze(-1)
        coefSpecular = coefSpecular.unsqueeze(-1)

    coefDiffuse = torch.clamp(coefDiffuse, min=0, max=1000)
    coefSpecular = torch.clamp(coefSpecular, min=0, max=1000)

    diffScaled = coefDiffuse.expand_as(diffOrig) * diffOrig
    specScaled = coefSpecular.expand_as(specOrig) * specOrig

    # Do the regression twice to avoid clamping
    renderedImg = torch.clamp(diffScaled + specScaled, 0, 1)
    renderedImg = renderedImg.reshape(nb, -1)
    imOrig = imOrig.reshape(nb, -1)

    # the same stuff as in LSregress
    coefIm = (
        torch.sum(renderedImg * imOrig, dim=1)
        / torch.clamp(torch.sum(renderedImg * renderedImg, dim=1), min=1e-5)
    ).detach()
    coefIm = torch.clamp(coefIm, 0.001, 1000)

    coefIm = coefIm.view(nb, 1, 1, 1)

    diffScaled = coefIm * diffScaled
    specScaled = coefIm * specScaled

    return diffScaled, specScaled


class encoder0(nn.Module):
    def __init__(self, cascadeLevel: int = 0, isSeg: bool = False):
        super(encoder0, self).__init__()
        self.isSeg = isSeg
        self.cascadeLevel = cascadeLevel

        self.pad1 = nn.ReplicationPad2d(1)
        if self.cascadeLevel == 0:
            self.conv1 = nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=4, stride=2, bias=True
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels=17, out_channels=64, kernel_size=4, stride=2, bias=True
            )

        self.gn1 = nn.GroupNorm(num_groups=4, num_channels=64)

        self.pad2 = nn.ZeroPad2d(1)
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=4, stride=2, bias=True
        )
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=128)

        self.pad3 = nn.ZeroPad2d(1)
        self.conv3 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=4, stride=2, bias=True
        )
        self.gn3 = nn.GroupNorm(num_groups=16, num_channels=256)

        self.pad4 = nn.ZeroPad2d(1)
        self.conv4 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=4, stride=2, bias=True
        )
        self.gn4 = nn.GroupNorm(num_groups=16, num_channels=256)

        self.pad5 = nn.ZeroPad2d(1)
        self.conv5 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=4, stride=2, bias=True
        )
        self.gn5 = nn.GroupNorm(num_groups=32, num_channels=512)

        self.pad6 = nn.ZeroPad2d(1)
        self.conv6 = nn.Conv2d(
            in_channels=512, out_channels=1024, kernel_size=3, stride=1, bias=True
        )
        self.gn6 = nn.GroupNorm(num_groups=64, num_channels=1024)

    def forward(self, x: torch.Tensor) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        x1 = F.relu(self.gn1(self.conv1(self.pad1(x))), True)
        x2 = F.relu(self.gn2(self.conv2(self.pad2(x1))), True)
        x3 = F.relu(self.gn3(self.conv3(self.pad3(x2))), True)
        x4 = F.relu(self.gn4(self.conv4(self.pad4(x3))), True)
        x5 = F.relu(self.gn5(self.conv5(self.pad5(x4))), True)
        x6 = F.relu(self.gn6(self.conv6(self.pad6(x5))), True)

        return x1, x2, x3, x4, x5, x6


class decoder0(nn.Module):
    def __init__(self, mode: Literal[0, 1, 2, 3, 4] = 0):
        super(decoder0, self).__init__()
        self.mode = mode

        self.dconv1 = nn.Conv2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.dgn1 = nn.GroupNorm(num_groups=32, num_channels=512)

        self.dconv2 = nn.Conv2d(
            in_channels=1024,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.dgn2 = nn.GroupNorm(num_groups=16, num_channels=256)

        self.dconv3 = nn.Conv2d(
            in_channels=512,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.dgn3 = nn.GroupNorm(num_groups=16, num_channels=256)

        self.dconv4 = nn.Conv2d(
            in_channels=512,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.dgn4 = nn.GroupNorm(num_groups=8, num_channels=128)

        self.dconv5 = nn.Conv2d(
            in_channels=256,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.dgn5 = nn.GroupNorm(num_groups=4, num_channels=64)

        self.dconv6 = nn.Conv2d(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.dgn6 = nn.GroupNorm(num_groups=4, num_channels=64)

        self.dpadFinal = nn.ReplicationPad2d(1)
        self.dconvFinal = nn.Conv2d(
            in_channels=64, out_channels=3, kernel_size=3, stride=1, bias=True
        )

    def forward(
        self,
        im: torch.Tensor,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
        x5: torch.Tensor,
        x6: torch.Tensor,
    ) -> torch.Tensor:
        dx1 = F.relu(self.dgn1(self.dconv1(x6)))

        xin1 = torch.cat([dx1, x5], dim=1)
        dx2 = F.relu(
            self.dgn2(
                self.dconv2(F.interpolate(xin1, scale_factor=2, mode="bilinear"))
            ),
            True,
        )

        if dx2.size(3) != x4.size(3) or dx2.size(2) != x4.size(2):
            dx2 = F.interpolate(dx2, [x4.size(2), x4.size(3)], mode="bilinear")
        xin2 = torch.cat([dx2, x4], dim=1)
        dx3 = F.relu(
            self.dgn3(
                self.dconv3(F.interpolate(xin2, scale_factor=2, mode="bilinear"))
            ),
            True,
        )

        if dx3.size(3) != x3.size(3) or dx3.size(2) != x3.size(2):
            dx3 = F.interpolate(dx3, [x3.size(2), x3.size(3)], mode="bilinear")
        xin3 = torch.cat([dx3, x3], dim=1)
        dx4 = F.relu(
            self.dgn4(
                self.dconv4(F.interpolate(xin3, scale_factor=2, mode="bilinear"))
            ),
            True,
        )

        if dx4.size(3) != x2.size(3) or dx4.size(2) != x2.size(2):
            dx4 = F.interpolate(dx4, [x2.size(2), x2.size(3)], mode="bilinear")
        xin4 = torch.cat([dx4, x2], dim=1)
        dx5 = F.relu(
            self.dgn5(
                self.dconv5(F.interpolate(xin4, scale_factor=2, mode="bilinear"))
            ),
            True,
        )

        if dx5.size(3) != x1.size(3) or dx5.size(2) != x1.size(2):
            dx5 = F.interpolate(dx5, [x1.size(2), x1.size(3)], mode="bilinear")
        xin5 = torch.cat([dx5, x1], dim=1)
        dx6 = F.relu(
            self.dgn6(
                self.dconv6(F.interpolate(xin5, scale_factor=2, mode="bilinear"))
            ),
            True,
        )

        if dx6.size(3) != im.size(3) or dx6.size(2) != im.size(2):
            dx6 = F.interpolate(dx6, [im.size(2), im.size(3)], mode="bilinear")
        x_orig = self.dconvFinal(self.dpadFinal(dx6))

        if self.mode == 0:
            x_out = torch.clamp(1.01 * torch.tanh(x_orig), -1, 1)
        elif self.mode == 1:
            x_orig = torch.clamp(1.01 * torch.tanh(x_orig), -1, 1)
            norm = torch.sqrt(torch.sum(x_orig * x_orig, dim=1).unsqueeze(1)).expand_as(
                x_orig
            )
            x_out = x_orig / torch.clamp(norm, min=1e-6)
        elif self.mode == 2:
            x_orig = torch.clamp(1.01 * torch.tanh(x_orig), -1, 1)
            x_out = torch.mean(x_orig, dim=1).unsqueeze(1)
        elif self.mode == 3:
            x_out = F.softmax(x_orig, dim=1)
        elif self.mode == 4:
            x_orig = torch.mean(x_orig, dim=1).unsqueeze(1)
            x_out = torch.clamp(1.01 * torch.tanh(x_orig), -1, 1)
        else:
            raise RuntimeError(f"Unknown mode {self.mode}")
        return x_out


class encoderLight(nn.Module):
    def __init__(self, SGNum: int, cascadeLevel: int = 0):
        super(encoderLight, self).__init__()

        self.cascadeLevel = cascadeLevel
        self.SGNum = SGNum

        self.preProcess = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(
                in_channels=11, out_channels=32, kernel_size=4, stride=2, bias=True
            ),
            nn.GroupNorm(num_groups=2, num_channels=32),
            nn.ReLU(inplace=True),
            nn.ZeroPad2d(1),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=4, stride=2, bias=True
            ),
            nn.GroupNorm(num_groups=4, num_channels=64),
            nn.ReLU(inplace=True),
        )

        self.pad1 = nn.ReplicationPad2d(1)
        if self.cascadeLevel == 0:
            self.conv1 = nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=4, stride=2, bias=True
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels=64 + SGNum * 7,
                out_channels=128,
                kernel_size=4,
                stride=2,
                bias=True,
            )
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=128)

        self.pad2 = nn.ZeroPad2d(1)
        self.conv2 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=4, stride=2, bias=True
        )
        self.gn2 = nn.GroupNorm(num_groups=16, num_channels=256)

        self.pad3 = nn.ZeroPad2d(1)
        self.conv3 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=4, stride=2, bias=True
        )
        self.gn3 = nn.GroupNorm(num_groups=16, num_channels=256)

        self.pad4 = nn.ZeroPad2d(1)
        self.conv4 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=4, stride=2, bias=True
        )
        self.gn4 = nn.GroupNorm(num_groups=32, num_channels=512)

        self.pad5 = nn.ZeroPad2d(1)
        self.conv5 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=4, stride=2, bias=True
        )
        self.gn5 = nn.GroupNorm(num_groups=32, num_channels=512)

        self.pad6 = nn.ZeroPad2d(1)
        self.conv6 = nn.Conv2d(
            in_channels=512, out_channels=1024, kernel_size=3, stride=1, bias=True
        )
        self.gn6 = nn.GroupNorm(num_groups=64, num_channels=1024)

    def forward(
        self, inputBatch: torch.Tensor, envs: Union[torch.Tensor, None] = None
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:

        input1: torch.Tensor = self.preProcess(inputBatch)
        input2 = envs

        if self.cascadeLevel == 0:
            x = input1
        else:
            assert input2 is not None
            x = torch.cat([input1, input2], dim=1)

        x1 = F.relu(self.gn1(self.conv1(self.pad1(x))), True)
        x2 = F.relu(self.gn2(self.conv2(self.pad2(x1))), True)
        x3 = F.relu(self.gn3(self.conv3(self.pad3(x2))), True)
        x4 = F.relu(self.gn4(self.conv4(self.pad4(x3))), True)
        x5 = F.relu(self.gn5(self.conv5(self.pad5(x4))), True)
        x6 = F.relu(self.gn6(self.conv6(self.pad6(x5))), True)

        return x1, x2, x3, x4, x5, x6


class decoderLight(nn.Module):
    def __init__(self, SGNum: int, mode: int = 0):
        super(decoderLight, self).__init__()

        self.SGNum = SGNum

        self.dconv1 = nn.Conv2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.dgn1 = nn.GroupNorm(num_groups=32, num_channels=512)

        self.dconv2 = nn.Conv2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.dgn2 = nn.GroupNorm(num_groups=32, num_channels=512)

        self.dconv3 = nn.Conv2d(
            in_channels=1024,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.dgn3 = nn.GroupNorm(num_groups=16, num_channels=256)

        self.dconv4 = nn.Conv2d(
            in_channels=512,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.dgn4 = nn.GroupNorm(num_groups=16, num_channels=256)

        self.dconv5 = nn.Conv2d(
            in_channels=512,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.dgn5 = nn.GroupNorm(num_groups=8, num_channels=128)

        self.dconv6 = nn.Conv2d(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.dgn6 = nn.GroupNorm(num_groups=8, num_channels=128)

        self.dpadFinal = nn.ReplicationPad2d(1)

        if mode == 0 or mode == 2:
            self.dconvFinal = nn.Conv2d(
                in_channels=128,
                out_channels=3 * SGNum,
                kernel_size=3,
                stride=1,
                bias=True,
            )
        elif mode == 1:
            self.dconvFinal = nn.Conv2d(
                in_channels=128, out_channels=SGNum, kernel_size=3, stride=1, bias=True
            )

        self.mode = mode

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
        x5: torch.Tensor,
        x6: torch.Tensor,
        env: torch.Tensor,
    ) -> torch.Tensor:
        dx1 = F.relu(self.dgn1(self.dconv1(x6)))

        xin1 = torch.cat([dx1, x5], dim=1)
        dx2 = F.relu(
            self.dgn2(
                self.dconv2(F.interpolate(xin1, scale_factor=2, mode="bilinear"))
            ),
            True,
        )

        if dx2.size(3) != x4.size(3) or dx2.size(2) != x4.size(2):
            dx2 = F.interpolate(dx2, [x4.size(2), x4.size(3)], mode="bilinear")
        xin2 = torch.cat([dx2, x4], dim=1)
        dx3 = F.relu(
            self.dgn3(
                self.dconv3(F.interpolate(xin2, scale_factor=2, mode="bilinear"))
            ),
            True,
        )

        if dx3.size(3) != x3.size(3) or dx3.size(2) != x3.size(2):
            dx3 = F.interpolate(dx3, [x3.size(2), x3.size(3)], mode="bilinear")
        xin3 = torch.cat([dx3, x3], dim=1)
        dx4 = F.relu(
            self.dgn4(
                self.dconv4(F.interpolate(xin3, scale_factor=2, mode="bilinear"))
            ),
            True,
        )

        if dx4.size(3) != x2.size(3) or dx4.size(2) != x2.size(2):
            dx4 = F.interpolate(dx4, [x2.size(2), x2.size(3)], mode="bilinear")
        xin4 = torch.cat([dx4, x2], dim=1)
        dx5 = F.relu(
            self.dgn5(
                self.dconv5(F.interpolate(xin4, scale_factor=2, mode="bilinear"))
            ),
            True,
        )

        if dx5.size(3) != x1.size(3) or dx5.size(2) != x1.size(2):
            dx5 = F.interpolate(dx5, [x1.size(2), x1.size(3)], mode="bilinear")
        xin5 = torch.cat([dx5, x1], dim=1)
        dx6 = F.relu(
            self.dgn6(
                self.dconv6(F.interpolate(xin5, scale_factor=2, mode="bilinear"))
            ),
            True,
        )

        if dx6.size(3) != env.size(3) or dx6.size(2) != env.size(2):
            dx6 = F.interpolate(dx6, [env.size(2), env.size(3)], mode="bilinear")
        x_orig = self.dconvFinal(self.dpadFinal(dx6))

        x_out = 1.01 * torch.tanh(self.dconvFinal(self.dpadFinal(dx6)))

        if self.mode == 1 or self.mode == 2:
            x_out = 0.5 * (x_out + 1)
            x_out = torch.clamp(x_out, 0, 1)
        elif self.mode == 0:
            bn, _, row, col = x_out.size()
            x_out = x_out.view(bn, self.SGNum, 3, row, col)
            x_out = x_out / torch.clamp(
                torch.sqrt(torch.sum(x_out * x_out, dim=2).unsqueeze(2)), min=1e-6
            ).expand_as(x_out)
        return x_out


class output2env:
    def __init__(self, SGNum: int, envWidth: int = 16, envHeight: int = 8):
        self.envWidth = envWidth
        self.envHeight = envHeight

        Az = ((np.arange(envWidth) + 0.5) / envWidth - 0.5) * 2 * np.pi
        El = ((np.arange(envHeight) + 0.5) / envHeight) * np.pi / 2.0
        Az, El = np.meshgrid(Az, El)
        Az = Az[np.newaxis, :, :]
        El = El[np.newaxis, :, :]
        lx = np.sin(El) * np.cos(Az)
        ly = np.sin(El) * np.sin(Az)
        lz = np.cos(El)
        ls = np.concatenate((lx, ly, lz), axis=0)
        ls = ls[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, :, :]
        self.ls = Variable(torch.from_numpy(ls.astype(np.float32)))

        self.SGNum = SGNum

        self.ls.requires_grad = False

    def to(self, device: torch.device) -> None:
        self.ls = self.ls.to(device)

    def fromSGtoIm(self, axis: torch.Tensor, lamb: torch.Tensor, weight: torch.Tensor):
        bn = axis.size(0)
        envRow, envCol = weight.size(2), weight.size(3)

        # Turn SG parameters to environmental maps
        axis = axis.unsqueeze(-1).unsqueeze(-1)

        weight = weight.view(bn, self.SGNum, 3, envRow, envCol, 1, 1)
        lamb = lamb.view(bn, self.SGNum, 1, envRow, envCol, 1, 1)

        mi = lamb.expand(
            [bn, self.SGNum, 1, envRow, envCol, self.envHeight, self.envWidth]
        ) * (
            torch.sum(
                axis.expand(
                    [bn, self.SGNum, 3, envRow, envCol, self.envHeight, self.envWidth]
                )
                * self.ls.expand(
                    [bn, self.SGNum, 3, envRow, envCol, self.envHeight, self.envWidth]
                ),
                dim=2,
            ).unsqueeze(2)
            - 1
        )
        envmaps = weight.expand(
            [bn, self.SGNum, 3, envRow, envCol, self.envHeight, self.envWidth]
        ) * torch.exp(mi).expand(
            [bn, self.SGNum, 3, envRow, envCol, self.envHeight, self.envWidth]
        )

        envmaps = torch.sum(envmaps, dim=1)

        return envmaps

    def output2env(
        self, axisOrig: torch.Tensor, lambOrig: torch.Tensor, weightOrig: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bn, _, envRow, envCol = weightOrig.size()

        axis = axisOrig

        weight = 0.999 * weightOrig
        weight = torch.tan(np.pi / 2 * weight)

        lambOrig = 0.999 * lambOrig
        lamb = torch.tan(np.pi / 2 * lambOrig)

        envmaps = self.fromSGtoIm(axis, lamb, weight)

        return envmaps, axis, lamb, weight


class renderingLayer:
    """
    A Pytorch-based rasterizing PBR renderer. This renderer assumes non-metalic materials.

    The renderer uses Lambertian BRDF for the diffuse component.

    The renderer uses

    Parameters
    ----------
    imWidth
        The width of the rendered images.
    imHeight
        The height of the rendered images.
    fov
        The field of view of the camera.
    F0
        The F0 parameter of the Schlick's approximation of the Fresnel-factor.
    envWidth
        The azimuth-resolution of the predicted environment maps.
    envHeight
        The elevation-resolution of the predicted environment maps.
    """

    def __init__(
        self,
        imWidth: int = 160,
        imHeight: int = 120,
        fov: float = 57,
        F0: float = 0.05,
        cameraPos: list[float] = [0, 0, 0],
        envWidth: int = 16,
        envHeight: int = 8,
    ):
        self.imHeight = imHeight
        self.imWidth = imWidth
        self.envWidth = envWidth
        self.envHeight = envHeight

        self.fov = fov / 180.0 * np.pi
        self.F0 = F0
        self.cameraPos = np.array(cameraPos, dtype=np.float32).reshape([1, 3, 1, 1])
        self.xRange = 1 * np.tan(self.fov / 2)
        self.yRange = float(imHeight) / float(imWidth) * self.xRange

        x, y = np.meshgrid(
            np.linspace(-self.xRange, self.xRange, imWidth),
            np.linspace(-self.yRange, self.yRange, imHeight),
        )
        y = np.flip(y, axis=0)
        z = -np.ones((imHeight, imWidth), dtype=np.float32)

        pCoord = np.stack([x, y, z]).astype(np.float32)
        self.pCoord = pCoord[np.newaxis, :, :, :]
        v = self.cameraPos - self.pCoord
        v = v / np.sqrt(np.maximum(np.sum(v * v, axis=1), 1e-12)[:, np.newaxis, :, :])
        v = v.astype(dtype=np.float32)

        self.v = Variable(torch.from_numpy(v))
        self.pCoord = Variable(torch.from_numpy(pCoord))

        self.up = torch.Tensor([0, 1, 0])

        # azimuth steps for the environment
        # +0.5 = half-texel alignment
        Az = ((np.arange(envWidth) + 0.5) / envWidth - 0.5) * 2 * np.pi
        # elevation steps
        # +0.5 = half-texel alignment
        El = ((np.arange(envHeight) + 0.5) / envHeight) * np.pi / 2.0

        # azimuth-grid, elevation-grid
        Az, El = np.meshgrid(Az, El)

        # convert the azimuth and elevation values to
        # direction vectors
        Az = Az.reshape(-1, 1)
        El = El.reshape(-1, 1)
        lx = np.sin(El) * np.cos(Az)
        ly = np.sin(El) * np.sin(Az)
        lz = np.cos(El)
        ls = np.concatenate(
            (lx, ly, lz), axis=1
        )  # (azimuth_step*elevation_step , dimension)

        R"""
        This is a somewhat technical variable for the diffuse color calculation.

        The implemented formula: $\sin(\text{elevation})*\pi^2/(w, h)$

        Parts:

        * division with the width and height: cache a division, that is going to be needed, because this tensor is used to approximate an integral with sum
        * $\sin(\text{elevation})$: equivalent to $\text{normal}\cdot\text{i}$
        * mulitplication with $\pi^2$: the approximated integral is taken on the surface of a sphere
        """
        envWeight = np.sin(El) * np.pi * np.pi / envWidth / envHeight

        self.ls = Variable(torch.from_numpy(ls.astype(np.float32)))
        self.envWeight = Variable(torch.from_numpy(envWeight.astype(np.float32)))
        self.envWeight = self.envWeight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    def to(self, device: torch.device) -> None:
        self.v = self.v.to(device)
        self.pCoord = self.pCoord.to(device)  # type: ignore
        self.up = self.up.to(device)
        self.ls = self.ls.to(device)
        self.envWeight = self.envWeight.to(device)

    def forwardEnv(
        self,
        diffusePred: torch.Tensor,
        normalPred: torch.Tensor,
        roughPred: torch.Tensor,
        envmap: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        diffusePred
            Format: ``ImLikes::AlbedoPred``
        normalPred
            Format: ``ImLikes::NormalPred``
        roughPred
            Format: ``ImLikes::RoughnessPred``
        envmap
            Format: ``EnvMap::EnvMapPred``
        """
        envR, envC = envmap.size(2), envmap.size(3)
        # envR = width of the image
        # envC = height of the image
        bn = diffusePred.size(0)
        # bn = batch size

        # do some preprocessing on the image (not that imporant)
        # filter diffuse color and normal
        diffusePred = F.adaptive_avg_pool2d(diffusePred, (envR, envC))
        normalPred = F.adaptive_avg_pool2d(normalPred, (envR, envC))
        # normalize predicted surface normals
        normalPred = normalPred / torch.sqrt(
            torch.clamp(torch.sum(normalPred * normalPred, dim=1), 1e-6, 1).unsqueeze(1)
        )
        # filter roughness
        roughPred = F.adaptive_avg_pool2d(roughPred, (envR, envC))

        temp = Variable(torch.FloatTensor(1, 1, 1, 1, 1))

        temp = temp.to(self.up.device)

        # lighting direction calculation (this is where the local coordinate system of the lighting is decided)
        # (you can see the equivalent shader-pseudoocode as a comment)
        ldirections = self.ls.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        # ldirections shape: (1, azimuth_step*elevation_step, dimension, 1, 1)
        # camyProj = self.up.dot(normalPred)*normalPred
        camyProj = (
            torch.einsum("b,abcd->acd", (self.up, normalPred))
            .unsqueeze(1)
            .expand_as(normalPred)
            * normalPred
        )
        # camy = normalize(self.up-camProj) //y orthogonal to the predicted normal vector and it is on the (up-normalPred plane)
        # // as a reminder, self.up = vec3(0, 1, 0)
        camy = F.normalize(
            self.up.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand_as(camyProj)
            - camyProj,
            dim=1,
        )
        # camx = -normalize(camy.cross(normalPred))
        camx = -F.normalize(torch.cross(camy, normalPred, dim=1), dim=1)

        # calculate the individual directions for each pixel of the environment map in world space
        l = (
            ldirections[:, :, 0:1, :, :] * camx.unsqueeze(1)
            + ldirections[:, :, 1:2, :, :] * camy.unsqueeze(1)
            + ldirections[:, :, 2:3, :, :] * normalPred.unsqueeze(1)
        )
        # end of calclation

        h = (self.v.unsqueeze(1) + l) / 2
        h = h / torch.sqrt(torch.clamp(torch.sum(h * h, dim=2), min=1e-6).unsqueeze(2))

        vdh = torch.sum((self.v * h), dim=2).unsqueeze(2)
        temp.data[0] = 2.0
        frac0 = self.F0 + (1 - self.F0) * torch.pow(
            temp.expand_as(vdh), (-5.55472 * vdh - 6.98316) * vdh
        )

        diffuseBatch = (diffusePred) / np.pi
        roughBatch = (roughPred + 1.0) / 2.0

        k = (roughBatch + 1) * (roughBatch + 1) / 8.0
        alpha = roughBatch * roughBatch
        alpha2 = alpha * alpha

        # n.dot(v)
        ndv = (
            torch.clamp(
                torch.sum(normalPred * self.v.expand_as(normalPred), dim=1), 0, 1
            )
            .unsqueeze(1)
            .unsqueeze(2)
        )
        # n.dot(h)
        ndh = torch.clamp(
            torch.sum(normalPred.unsqueeze(1) * h, dim=2), 0, 1
        ).unsqueeze(2)
        # n.dot(l)
        ndl = torch.clamp(
            torch.sum(normalPred.unsqueeze(1) * l, dim=2), 0, 1
        ).unsqueeze(2)

        # D_?
        # alpha2 = roughness
        frac = alpha2.unsqueeze(1).expand_as(frac0) * frac0
        nom0 = ndh * ndh * (alpha2.unsqueeze(1).expand_as(ndh) - 1) + 1
        nom1 = ndv * (1 - k.unsqueeze(1).expand_as(ndh)) + k.unsqueeze(1).expand_as(ndh)
        nom2 = ndl * (1 - k.unsqueeze(1).expand_as(ndh)) + k.unsqueeze(1).expand_as(ndh)
        nom = torch.clamp(4 * np.pi * nom0 * nom0 * nom1 * nom2, 1e-6, 4 * np.pi)
        specPred = frac / nom

        # some unimportant transforms
        envmap = envmap.view([bn, 3, envR, envC, self.envWidth * self.envHeight])
        envmap = envmap.permute([0, 4, 1, 2, 3])

        # diffuse radiance
        brdfDiffuse = diffuseBatch.unsqueeze(1).expand(
            [bn, self.envWidth * self.envHeight, 3, envR, envC]
        ) * ndl.expand([bn, self.envWidth * self.envHeight, 3, envR, envC])
        # irradiance = envmap * self.envWeight
        colorDiffuse = torch.sum(
            brdfDiffuse * envmap * self.envWeight.expand_as(brdfDiffuse), dim=1
        )

        # specular radiance
        brdfSpec = specPred.expand(
            [bn, self.envWidth * self.envHeight, 3, envR, envC]
        ) * ndl.expand([bn, self.envWidth * self.envHeight, 3, envR, envC])
        colorSpec = torch.sum(
            brdfSpec * envmap * self.envWeight.expand_as(brdfSpec), dim=1
        )

        return colorDiffuse, colorSpec


def BatchRankingLoss(
    albedoPred: torch.Tensor,
    eqPoint: np.ndarray,
    eqWeight: np.ndarray,
    darkerPoint: np.ndarray,
    darkerWeight: np.ndarray,
):
    tau = 0.5
    height, width = albedoPred.size(1), albedoPred.size(2)

    reflectance = torch.mean(albedoPred, dim=0)
    reflectLog = torch.log(reflectance + 0.001)
    reflectLog = reflectLog.view(-1)

    eqPointTensor = Variable(torch.from_numpy(eqPoint).long()).cuda()
    eqWeightTensor = Variable(torch.from_numpy(eqWeight).float()).cuda()
    darkerPointTensor = Variable(torch.from_numpy(darkerPoint).long()).cuda()
    darkerWeightTensor = Variable(torch.from_numpy(darkerWeight).float()).cuda()

    # compute the eq loss
    r1, c1, r2, c2 = torch.split(eqPointTensor, 1, dim=1)
    p1 = (r1 * width + c1).view(-1)
    p1.requires_grad = False
    p2 = (r2 * width + c2).view(-1)
    p2.requires_grad = False
    rf1 = torch.index_select(reflectLog, 0, p1)
    rf2 = torch.index_select(reflectLog, 0, p2)
    eqWeightTensor = eqWeightTensor.view(-1)

    eqLoss = torch.mean(eqWeightTensor * torch.pow(rf1 - rf2, 2))

    # compute the darker loss
    r1, c1, r2, c2 = torch.split(darkerPointTensor, 1, dim=1)
    p1 = (r1 * width + c1).view(-1)
    p1.requires_grad = False
    p2 = (r2 * width + c2).view(-1)
    p2.requires_grad = False
    rf1 = torch.index_select(reflectLog, 0, p1)
    rf2 = torch.index_select(reflectLog, 0, p2)
    darkerWeightTensor = darkerWeightTensor.view(-1)

    darkerLoss = torch.mean(darkerWeightTensor * torch.pow(F.relu(rf2 - rf1 + tau), 2))

    return eqLoss, darkerLoss
