template = 'sbatch slurm.sh config/{ds}/{name}.json {name} 30 --model_checkpoint_period 1 --prediction_checkpoint_period 1 --monitor val_Dice'

name_template = '{ds}_f{fold}_2d_{modality}{aug}'

folds = [0,1,2,3,4]
augs = ['', '_aug']
modalities = ['CECT', 'LDCT', 'PET',
            'PET_CECT',
            'PET_LDCT',
            'T2W',
            'ADC',
            'T2W_CECT',
            'T2W_LDCT',
            'T2W_ADC',
            'PET_T2W',
            'PET_CECT_T2W',
            'PET_LDCT_T2W' ]


ds = 36
for aug in augs:
    for modality in modalities:
        for fold in folds:
            name = name_template.format(ds=ds, fold=fold, modality=modality, aug=aug)
            print(template.format(ds=ds, name=name))

ds = 86
for aug in augs:
    for modality in modalities[:5]:
        for fold in folds:
            name = name_template.format(ds=ds, fold=fold, modality=modality, aug=aug)
            print(template.format(ds=ds, name=name))
