import splitfolders

dir_input = r"C:\Doctorado\Neurociencia\CASME\CASME_3\part_A\dataset"
splitfolders.ratio(dir_input,
            output=f'{dir_input}-splitted',
            seed=1337, ratio=(.8, .1, .1))
