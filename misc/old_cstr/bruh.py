import numpy as np

P_discr = np.array(
    [
        [13.26218061, 6.84863163, 3.73771796, 1.22134438],
        [6.84863163, 6.70881811, 1.84128064, 0.59959977],
        [3.73771796, 1.84128064, 2.2301409, 0.54665047],
        [1.22134438, 0.59959977, 0.54665047, 0.45878073],
    ]
)

P_cont = np.array(
    [
        [263.11004328, 137.10970469, 75.27514156, 24.21511573],
        [137.10970469, 123.63976555, 37.0769089, 11.88693747],
        [75.27514156, 37.0769089, 39.26890225, 11.14075444],
        [24.21511573, 11.88693747, 11.14075444, 6.76533351],
    ]
)

P_matlab = np.array(
    [
        [
            1.4646778374584373,
            0.6676889516721198,
            0.35446715117028615,
            0.1032442200508634,
        ],
        [
            0.6676889516721198,
            1.407812935783267,
            0.17788030743777067,
            0.050059833257226405,
        ],
        [
            0.35446715117028615,
            0.1778803074377706,
            0.6336052592712396,
            0.01110329497282364,
        ],
        [
            0.10324422005086348,
            0.05005983325722643,
            0.011103294972823655,
            0.229412393739723,
        ],
    ]
)

print(np.divide(P_discr, P_cont))
print(np.divide(P_discr, P_matlab))
print(np.divide(P_cont, P_matlab))
