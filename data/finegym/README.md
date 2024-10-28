# Setting up FineGym

This directory contains the splits and labels converted from FineGym: https://sdolivia.github.io/FineGym/.

To download the trimmed videos frames and generate the folder structure for frames, follow the instructions provided in E2E-Spot: https://github.com/jhong93/spot/tree/main/data/finegym

Frames are extracted at a resolution of 224x224, following ths frame naming convention:

```
data-folder
└───FineGymFrames
    └───0jqn1vxdhls
        |frame3951.jpg
        |frame3952.jpg
        |...
        |frame141058.jpg
    └───0MtilFKz4cA
        |frame5925.jpg
        |frame5926.jpg
        |...
```

---

## License from FineGym

Creative Commons Attribution-NonCommercial 4.0 International License
(See https://sdolivia.github.io/FineGym/)

## License from E2E-Spot

Copyright 2022 James Hong, Haotian Zhang, Matthew Fisher, Michael Gharbi,
Kayvon Fatahalian

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.