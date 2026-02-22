import numpy as np
import re
import pandas as pd


def preprocess_data(df:pd.DataFrame):
    df = df.copy()

    df['TouchScreen'] = df['ScreenResolution'].str.contains('Touchscreen').astype(int)
    df['IPS'] = df['ScreenResolution'].str.contains('IPS').astype(int)

    df[['Width','Height']] = df['ScreenResolution'].str.extract(r'(\d+)x(\d+)').astype(int)
    df.drop(columns=['ScreenResolution'], inplace=True)

    df['PPI'] = (((df['Width']**2 + df['Height']**2)**0.5)/df['Inches']).round(2)
    df.drop(columns=['Inches','Width','Height'], inplace=True)

    def cpu_brand(cpu):
        cpu = cpu.lower()
        if "i3" in cpu: return 3
        elif "i5" in cpu: return 5
        elif "i7" in cpu: return 7
        elif "intel" in cpu: return 1
        elif "amd" in cpu: return 2
        else: return 0

    df["Cpu_Brand"] = df["Cpu"].apply(cpu_brand)
    df.drop(columns=['Cpu'], inplace=True)

    df['Ram'] = df['Ram'].str.replace('GB','').astype(int)
    df['Weight'] = df['Weight'].str.replace('kg','').astype(float)

    df['Memory'] = df['Memory'].astype(str).replace('\.0','', regex=True)
    df['Memory'] = df['Memory'].str.replace('GB','', regex=False)
    df['Memory'] = df['Memory'].str.replace('TB','000', regex=False)

    mem = df['Memory'].str.split("+", expand=True)
    df['first'] = mem[0].str.extract(r'(\d+)').fillna(0).astype(int)
    df['second'] = mem[1].str.extract(r'(\d+)').fillna(0).astype(int)

    def extract_types(x, t):
        return int(t in str(x).upper())

    for t in ['SSD','HDD','FLASH','HYBRID']:
        df[f'{t}_1'] = mem[0].apply(lambda x: extract_types(x,t))
        df[f'{t}_2'] = mem[1].apply(lambda x: extract_types(x,t))

    df["HDD"] = df["first"]*df["HDD_1"] + df["second"]*df["HDD_2"]
    df["SSD"] = df["first"]*df["SSD_1"] + df["second"]*df["SSD_2"]

    df.drop(columns=df.filter(regex='_1|_2|first|second|Memory').columns, inplace=True)

    gpu_map = {'Intel':0,'AMD':1,'Nvidia':2}
    df['Gpu'] = df['Gpu'].str.split().str[0].map(gpu_map).fillna(0)

    df['OpSys'] = np.select(
        [df['OpSys'].str.contains('Windows',case=False),
         df['OpSys'].str.contains('Mac',case=False)],
        [0,1],
        default=2
    )

    return df



def preprocess(df):
    df['TouchScreen'] = df['ScreenResolution'].apply(lambda x:1 
                                                      if 'Touchscreen' in x else 0)

    df['IPS'] = df['ScreenResolution'].apply(lambda x:1 
                                                        if 'IPS' in x else 0)

    df['Resolution'] = df['ScreenResolution'].str.extract(r'(\d+x\d+)')
    df[['Width','Height']] = (df['Resolution'].str.extract(r'(\d+)x(\d+)').astype(int))
    df.drop(columns=['Resolution','ScreenResolution'],axis=1,inplace=True)

    df['PPI'] = (((df['Width']**2 + df['Height']**2) ** 0.5) / df['Inches']).round(2)
    df.drop(columns=['Inches','Width','Height'],inplace=True)

    df['Cpu_Brand'] = np.where(
        df['Cpu'].str.lower().str.contains(r'intel core i3'), 'Intel Core i3',
        np.where(
            df['Cpu'].str.lower().str.contains(r'intel core i5'), 'Intel Core i5',
            np.where(
                df['Cpu'].str.lower().str.contains(r'intel core i7'), 'Intel Core i7',
                np.where(
                    df['Cpu'].str.lower().str.contains(r'intel'), 'Other Intel',
                    np.where(
                        df['Cpu'].str.lower().str.contains(r'amd'), 'AMD',
                        'Other'
                    )
                )
            )
        )
    )
    cpu_map = {
    'Intel Core i3': 3,
    'Intel Core i5': 5,
    'Intel Core i7': 7,
    'Other Intel': 1,
    'AMD': 2,
    'Other': 0
    }

    df['Cpu_Brand'] = df['Cpu_Brand'].map(cpu_map)
    df.drop(columns=['Cpu'], inplace=True)


    df['Ram'] = df['Ram'].str.replace('GB','').str.strip()
    df['Weight'] = df['Weight'].str.replace('kg','').str.strip()

    df['Ram'] = df['Ram'].astype('int32')
    df['Weight'] = df['Weight'].astype('float32')

    import re

    # Step 1: Clean Memory column
    df['Memory'] = df['Memory'].astype(str).replace('\.0','', regex=True)
    df['Memory'] = df['Memory'].str.replace('GB','', regex=False)
    df['Memory'] = df['Memory'].str.replace('TB','000', regex=False)

    # Step 2: Split combo storage
    memory_split = df['Memory'].str.split("+", n=1, expand=True).apply(lambda x: x.str.strip())
    df['first_split'] = memory_split[0]
    df['second_split'] = memory_split[1].fillna("")  # empty string if no second part

    # Step 3: Extract numeric values
    df['first'] = df['first_split'].str.extract(r'(\d+)').astype(int)
    df['second'] = df['second_split'].str.extract(r'(\d+)').fillna(0).astype(int)

    # Max and total memory
    # df['Memory_Num_Max'] = df[['Memory_Num_1','Memory_Num_2']].max(axis=1)
    # df['Memory_Num_Total'] = df[['Memory_Num_1','Memory_Num_2']].sum(axis=1)

    # Step 4: Extract types separately
    def extract_memory_types(mem_str):
        types = re.findall(r'(SSD|HDD|FLASH|HYBRID)', mem_str.upper())
        return types if types else ['Other']

    # First split types
    types_1 = df['first_split'].apply(extract_memory_types)
    for t in ['SSD','HDD','FLASH','HYBRID']:
        df[f'Memory_1_{t}'] = types_1.apply(lambda x: int(t in x))

    # Second split types
    types_2 = df['second_split'].apply(extract_memory_types)
    for t in ['SSD','HDD','FLASH','HYBRID']:
        df[f'Memory_2_{t}'] = types_2.apply(lambda x: int(t in x))

    # Step 5: Drop temporary columns if needed
    df.drop(columns=['first_split','second_split'], inplace=True)

        # Create final memory type numeric columns
    df["HDD"]=(df["first"]*df["Memory_1_HDD"] + df["second"]*df["Memory_2_HDD"])
    df["SSD"]=(df["first"]*df["Memory_1_SSD"] + df["second"]*df["Memory_2_SSD"])
    df["Hybrid"]=(df["first"]*df["Memory_1_HYBRID"] + df["second"]*df["Memory_2_HYBRID"])
    df["Flash_St"]=(df["first"]*df["Memory_1_FLASH"] +df["second"]*df["Memory_2_FLASH"])

    # Drop unnecessary columns
    df.drop(columns=['first', 'second', 'Memory','Hybrid','Flash_St',
                    'Memory_1_HDD', 'Memory_2_HDD', 
                    'Memory_1_SSD','Memory_2_SSD',
                    'Memory_1_HYBRID', 'Memory_2_HYBRID', 
                    'Memory_1_FLASH', 'Memory_2_FLASH'
                    ],
                    inplace=True)
    df['Gpu'] = df['Gpu'].str.split().str[0]
    gpu_map = {
    'Intel': 0,
    'AMD': 1,
    'Nvidia': 2
    }

    df['Gpu'] = df['Gpu'].map(gpu_map)
    df['OpSys'] = np.where(df['OpSys'].str.contains('Windows', case=False), 'Windows',
                     np.where(df['OpSys'].str.contains('Mac', case=False), 'Mac',
                              'Other'))
    os_map = {
    'Windows': 0,
    'Mac': 1,
    'Other': 2
    }

    df['OpSys'] = df['OpSys'].map(os_map)
    df['Price'] = np.log1p(df['Price'])
    df = pd.get_dummies(df, columns=['Company','TypeName'])

    return df


    

