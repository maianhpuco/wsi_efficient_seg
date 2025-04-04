
## Dataset Directory Structure (Image file counts per folder)
```
kidney_pathology_image
├── test
│   ├── Task1_patch_level
│   │   └── test
│   │       ├── 56Nx
│   │       │   ├── 12-299
│   │       │   │   ├── img
│   │       │   │   │   └── [Image Files: 159]
│   │       │   │   └── mask
│   │       │   │       └── [Image Files: 159]
│   │       │   ├── 12-300
│   │       │   │   ├── img
│   │       │   │   │   └── [Image Files: 165]
│   │       │   │   └── mask
│   │       │   │       └── [Image Files: 165]
│   │       │   └── 12-301
│   │       │       ├── img
│   │       │       │   └── [Image Files: 139]
│   │       │       └── mask
│   │       │           └── [Image Files: 139]
│   │       ├── DN
│   │       │   ├── 11-362
│   │       │   │   ├── img
│   │       │   │   │   └── [Image Files: 154]
│   │       │   │   └── mask
│   │       │   │       └── [Image Files: 154]
│   │       │   ├── 11-363
│   │       │   │   ├── img
│   │       │   │   │   └── [Image Files: 113]
│   │       │   │   └── mask
│   │       │   │       └── [Image Files: 113]
│   │       │   └── 11-364
│   │       │       ├── img
│   │       │       │   └── [Image Files: 124]
│   │       │       └── mask
│   │       │           └── [Image Files: 124]
│   │       ├── NEP25
│   │       │   ├── 18-578
│   │       │   │   ├── img
│   │       │   │   │   └── [Image Files: 100]
│   │       │   │   └── mask
│   │       │   │       └── [Image Files: 100]
│   │       │   ├── 18-579
│   │       │   │   ├── img
│   │       │   │   │   └── [Image Files: 105]
│   │       │   │   └── mask
│   │       │   │       └── [Image Files: 105]
│   │       │   └── 18-580
│   │       │       ├── img
│   │       │       │   └── [Image Files: 80]
│   │       │       └── mask
│   │       │           └── [Image Files: 80]
│   │       └── normal
│   │           ├── normal_M1574
│   │           │   ├── img
│   │           │   │   └── [Image Files: 459]
│   │           │   └── mask
│   │           │       └── [Image Files: 459]
│   │           ├── normal_M1580
│   │           │   ├── img
│   │           │   │   └── [Image Files: 350]
│   │           │   └── mask
│   │           │       └── [Image Files: 350]
│   │           └── normal_M1581
│   │               ├── img
│   │               │   └── [Image Files: 357]
│   │               └── mask
│   │                   └── [Image Files: 357]
│   └── Task2_WSI_level
│       ├── 56NX
│       │   └── [Image Files: 6]
│       ├── DN
│       │   └── [Image Files: 6]
│       ├── NEP25
│       │   └── [Image Files: 6]
│       └── normal
│           └── [Image Files: 6]
├── train
│   ├── Task1_patch_level
│   │   └── train
│   │       ├── 56Nx
│   │       │   ├── 12_116
│   │       │   │   ├── img
│   │       │   │   │   └── [Image Files: 92]
│   │       │   │   └── mask
│   │       │   │       └── [Image Files: 92]
│   │       │   ├── 12_117
│   │       │   │   ├── img
│   │       │   │   │   └── [Image Files: 96]
│   │       │   │   └── mask
│   │       │   │       └── [Image Files: 96]
│   │       │   ├── 12_169
│   │       │   │   ├── img
│   │       │   │   │   └── [Image Files: 141]
│   │       │   │   └── mask
│   │       │   │       └── [Image Files: 141]
│   │       │   ├── 12_170
│   │       │   │   ├── img
│   │       │   │   │   └── [Image Files: 86]
│   │       │   │   └── mask
│   │       │   │       └── [Image Files: 86]
│   │       │   └── 12_171
│   │       │       ├── img
│   │       │       │   └── [Image Files: 143]
│   │       │       └── mask
│   │       │           └── [Image Files: 143]
│   │       ├── DN
│   │       │   ├── 11_356
│   │       │   │   ├── img
│   │       │   │   │   └── [Image Files: 135]
│   │       │   │   └── mask
│   │       │   │       └── [Image Files: 135]
│   │       │   ├── 11_357
│   │       │   │   ├── img
│   │       │   │   │   └── [Image Files: 134]
│   │       │   │   └── mask
│   │       │   │       └── [Image Files: 134]
│   │       │   ├── 11_358
│   │       │   │   ├── img
│   │       │   │   │   └── [Image Files: 188]
│   │       │   │   └── mask
│   │       │   │       └── [Image Files: 188]
│   │       │   ├── 11_367
│   │       │   │   ├── img
│   │       │   │   │   └── [Image Files: 129]
│   │       │   │   └── mask
│   │       │   │       └── [Image Files: 129]
│   │       │   └── 11_370
│   │       │       ├── img
│   │       │       │   └── [Image Files: 138]
│   │       │       └── mask
│   │       │           └── [Image Files: 138]
│   │       ├── NEP25
│   │       │   ├── 08_368_01
│   │       │   │   ├── img
│   │       │   │   │   └── [Image Files: 143]
│   │       │   │   └── mask
│   │       │   │       └── [Image Files: 143]
│   │       │   ├── 08_368_02
│   │       │   │   ├── img
│   │       │   │   │   └── [Image Files: 187]
│   │       │   │   └── mask
│   │       │   │       └── [Image Files: 187]
│   │       │   ├── 08_368_03
│   │       │   │   ├── img
│   │       │   │   │   └── [Image Files: 166]
│   │       │   │   └── mask
│   │       │   │       └── [Image Files: 166]
│   │       │   ├── 08_373_01
│   │       │   │   ├── img
│   │       │   │   │   └── [Image Files: 146]
│   │       │   │   └── mask
│   │       │   │       └── [Image Files: 146]
│   │       │   ├── 08_373_02
│   │       │   │   ├── img
│   │       │   │   │   └── [Image Files: 133]
│   │       │   │   └── mask
│   │       │   │       └── [Image Files: 133]
│   │       │   ├── 08_373_03
│   │       │   │   ├── img
│   │       │   │   │   └── [Image Files: 111]
│   │       │   │   └── mask
│   │       │   │       └── [Image Files: 111]
│   │       │   ├── 08_471_01
│   │       │   │   ├── img
│   │       │   │   │   └── [Image Files: 156]
│   │       │   │   └── mask
│   │       │   │       └── [Image Files: 156]
│   │       │   ├── 08_471_02
│   │       │   │   ├── img
│   │       │   │   │   └── [Image Files: 177]
│   │       │   │   └── mask
│   │       │   │       └── [Image Files: 177]
│   │       │   ├── 08_471_03
│   │       │   │   ├── img
│   │       │   │   │   └── [Image Files: 152]
│   │       │   │   └── mask
│   │       │   │       └── [Image Files: 152]
│   │       │   ├── 08_472_01
│   │       │   │   ├── img
│   │       │   │   │   └── [Image Files: 147]
│   │       │   │   └── mask
│   │       │   │       └── [Image Files: 147]
│   │       │   ├── 08_472_02
│   │       │   │   ├── img
│   │       │   │   │   └── [Image Files: 150]
│   │       │   │   └── mask
│   │       │   │       └── [Image Files: 150]
│   │       │   ├── 08_472_03
│   │       │   │   ├── img
│   │       │   │   │   └── [Image Files: 93]
│   │       │   │   └── mask
│   │       │   │       └── [Image Files: 93]
│   │       │   ├── 08_474_01
│   │       │   │   ├── img
│   │       │   │   │   └── [Image Files: 156]
│   │       │   │   └── mask
│   │       │   │       └── [Image Files: 156]
│   │       │   ├── 08_474_02
│   │       │   │   ├── img
│   │       │   │   │   └── [Image Files: 218]
│   │       │   │   └── mask
│   │       │   │       └── [Image Files: 218]
│   │       │   └── 08_474_03
│   │       │       ├── img
│   │       │       │   └── [Image Files: 128]
│   │       │       └── mask
│   │       │           └── [Image Files: 128]
│   │       └── normal
│   │           ├── normal_F1
│   │           │   ├── img
│   │           │   │   └── [Image Files: 358]
│   │           │   └── mask
│   │           │       └── [Image Files: 358]
│   │           ├── normal_F1576
│   │           │   ├── img
│   │           │   │   └── [Image Files: 390]
│   │           │   └── mask
│   │           │       └── [Image Files: 390]
│   │           ├── normal_F2
│   │           │   ├── img
│   │           │   │   └── [Image Files: 313]
│   │           │   └── mask
│   │           │       └── [Image Files: 313]
│   │           ├── normal_F3
│   │           │   ├── img
│   │           │   │   └── [Image Files: 370]
│   │           │   └── mask
│   │           │       └── [Image Files: 370]
│   │           └── normal_F4
│   │               ├── img
│   │               │   └── [Image Files: 355]
│   │               └── mask
│   │                   └── [Image Files: 355]
│   └── Task2_WSI_level
│       ├── 56Nx
│       │   └── [Image Files: 10]
│       ├── DN
│       │   └── [Image Files: 10]
│       ├── NEP25
│       │   └── [Image Files: 30]
│       └── normal
│           └── [Image Files: 10]
└── validation
    ├── Task1_patch_level
    │   └── validation
    │       ├── 56Nx
    │       │   ├── 12-173
    │       │   │   ├── img
    │       │   │   │   └── [Image Files: 144]
    │       │   │   └── mask
    │       │   │       └── [Image Files: 144]
    │       │   └── 12-174
    │       │       ├── img
    │       │       │   └── [Image Files: 130]
    │       │       └── mask
    │       │           └── [Image Files: 130]
    │       ├── DN
    │       │   ├── 11-359
    │       │   │   ├── img
    │       │   │   │   └── [Image Files: 182]
    │       │   │   └── mask
    │       │   │       └── [Image Files: 182]
    │       │   └── 11-361
    │       │       ├── img
    │       │       │   └── [Image Files: 117]
    │       │       └── mask
    │       │           └── [Image Files: 117]
    │       ├── NEP25
    │       │   ├── 18-575
    │       │   │   ├── img
    │       │   │   │   └── [Image Files: 106]
    │       │   │   └── mask
    │       │   │       └── [Image Files: 106]
    │       │   └── 18-577
    │       │       ├── img
    │       │       │   └── [Image Files: 103]
    │       │       └── mask
    │       │           └── [Image Files: 103]
    │       └── normal
    │           ├── normal_M1
    │           │   ├── img
    │           │   │   └── [Image Files: 415]
    │           │   └── mask
    │           │       └── [Image Files: 415]
    │           └── normal_M2
    │               ├── img
    │               │   └── [Image Files: 446]
    │               └── mask
    │                   └── [Image Files: 446]
    └── Task2_WSI_level
        ├── 56Nx
        │   └── [Image Files: 4]
        ├── DN
        │   └── [Image Files: 4]
        ├── NEP25
        │   └── [Image Files: 4]
        └── normal
            └── [Image Files: 4]
```
