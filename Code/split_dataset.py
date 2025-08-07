import splitfolders

dir_input = "C:/Doctorado/Neurociencia/CASME/CASME-II-Binary"
splitfolders.ratio(dir_input,
            output=f'{dir_input}-splitted',
            seed=1337, ratio=(.8, .1, .1))
