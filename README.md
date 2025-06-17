# GCPP7-3
GCPP7-3: Global Cloud Physical Properties with 0.07° and 3h resolution Inversion by CloudUNet

A long-term global cloud physical products with 0.07° spatial resolution and 3h temporal resolution from 2000 to 2022, covering longitudes from -180° to 180° and latitudes from -70° to 70°. The data include four variables: cloud phase (CLP), cloud top height (CTH), cloud optical thickness (COT), and cloud effective radius (CER).

Detailed information:

3h temporal resolution is UTC time 0, 3, 6, 9, 12, 15, 18, and 21 hours daily.

CLP: 0 clear, 1 liquid, 2 ice, missing -1.

CTH: CTH in km; missing: NaN; range [0, 18].

COT: COT in unitless; missing: NaN; range [0, 150].

CER: CER in µm; missing: NaN; range [0, 100].

 

The file names and directory structure are organized in the following format:

```

{year}.nc/{year}{month}{day}{hour}.nc

```

 

## change list:

V1: init commit.
