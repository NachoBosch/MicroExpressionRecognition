import h5py
import json

def fix_h5_model(input_path, output_path):
    """
    Arregla el modelo H5 convirtiendo axis de lista a entero en BatchNormalization
    """
    with h5py.File(input_path, 'r') as f_in:
        with h5py.File(output_path, 'w') as f_out:
            # Copiar todos los grupos y datasets
            for key in f_in.keys():
                f_in.copy(key, f_out)
            
            # Arreglar model_config
            if 'model_config' in f_in.attrs:
                model_config_str = f_in.attrs['model_config']
                if isinstance(model_config_str, bytes):
                    model_config_str = model_config_str.decode('utf-8')
                
                model_config = json.loads(model_config_str)
                
                # Función recursiva para arreglar axis
                def fix_axis_in_config(config):
                    if isinstance(config, dict):
                        # Arreglar BatchNormalization
                        if config.get('class_name') == 'BatchNormalization':
                            if 'config' in config and 'axis' in config['config']:
                                axis = config['config']['axis']
                                if isinstance(axis, list):
                                    config['config']['axis'] = axis[0] if len(axis) > 0 else 3
                                    print(f"✓ Arreglado BatchNormalization: {config['config'].get('name', 'unnamed')}")
                        
                        # Recursivamente revisar todos los valores
                        for key, value in config.items():
                            config[key] = fix_axis_in_config(value)
                    
                    elif isinstance(config, list):
                        return [fix_axis_in_config(item) for item in config]
                    
                    return config
                
                model_config = fix_axis_in_config(model_config)
                f_out.attrs['model_config'] = json.dumps(model_config)
            
            # Copiar otros atributos
            for key in f_in.attrs.keys():
                if key != 'model_config':
                    f_out.attrs[key] = f_in.attrs[key]
    
    print(f"\n✓ Modelo arreglado guardado en: {output_path}")

if __name__ == "__main__":
    input_model = 'casme_multiclass_model.h5'
    output_model = 'casme_multiclass_model_fixed.h5'
    
    print("Arreglando modelo...")
    fix_h5_model(input_model, output_model)
    print("¡Listo! Ahora usa 'casme_multiclass_model_fixed.h5' en tu código.")