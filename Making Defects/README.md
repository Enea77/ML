<h1> Instruction for making defects </h1>

To run `make_defects.py`, change line 161: <br />
`exp_clean = np.load(path/to/file)` <br />
The file path points to an image array saved as a `.npy` file <br />
And in lines 165, 168 and 171 change the file paths for `Ti_SPOT`, `Sr_SPOT` and `DARK_SPOT` <br />
to a `.npy` file of a Ti, Sr and Vacancy column respectively. <br />

The code will then produce 6 `.npy` files like the input image arrays for each of the following point defects: <br />
Ti Vacancy, <br />
Ti Anti site, <br />
Ti Ferromagnetic distorsion, <br />
Sr Vacancy, <br />
Sr Anti site, <br />
Sr Ferromagnetic distorsion. <br />
