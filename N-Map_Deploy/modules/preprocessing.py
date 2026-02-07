import pandas as pd
import re

def load_data(file):
    """Loads data from CSV or Excel file."""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            return None
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def bin_age(age):
    """Bins age into 10-year groups."""
    if pd.isna(age):
        return None
    try:
        age = int(age)
        if age < 10: return "0-9세"
        elif age < 20: return "10대"
        elif age < 30: return "20대"
        elif age < 40: return "30대"
        elif age < 50: return "40대"
        elif age < 60: return "50대"
        elif age < 70: return "60대"
        elif age < 80: return "70대"
        elif age < 90: return "80대"
        else: return "90세 이상"
    except:
        return None

def bin_surgery_time(minutes):
    """Bins surgery time into Short, Medium, Long."""
    if pd.isna(minutes):
        return None
    try:
        minutes = float(minutes)
        if minutes < 60:
            return "단기(<60분)"
        elif minutes <= 180:
            return "중기(60-180분)"
        else:
            return "장기(>180분)"
    except:
        return None

def preprocess_data(df):
    """
    Preprocesses the raw dataframe:
    - Bins 'Age' (연령)
    - Bins 'Surgery Time' (수술시간)
    - Explodes 'Nursing Interventions' (간호중재)
    """
    # 0. Copy Data
    data = df.copy()

    # 1. Wide Format Handling (User specific format)
    # Check for '간호중재1', '간호중재2'...
    intervention_cols = [c for c in data.columns if c.startswith('간호중재')]
    if intervention_cols:
        # Combine all intervention columns into a single list
        data['간호중재'] = data[intervention_cols].apply(lambda x: [str(i).strip() for i in x if pd.notna(i) and str(i).strip() != ''], axis=1)
        # Note: We don't need to explode here instantly if we want to preserve row structure first, 
        # but the current logic expects '간호중재' to be exploded later.
        # Let's join them with comma to match the "string" format expected by step 5, or just handle it directly.
        # Actually, step 5 expects a string. Let's make it a string list for now.
        data['간호중재_list'] = data['간호중재'] # Keep list version
        data['간호중재'] = data['간호중재'].apply(lambda x: ', '.join(x))

    # Check for Surgery Time Calculation
    if '절개시간' in data.columns and '봉합시간' in data.columns:
        # Calculate duration in minutes
        # Data format might be string "HH:MM" or datetime objects
        def calculate_duration(row):
            try:
                start_val = row['절개시간']
                end_val = row['봉합시간']
                
                # Helper to convert to datetime
                def to_dt(val):
                    if pd.isna(val): return None
                    if isinstance(val, pd.Timestamp): return val
                    if isinstance(val, (datetime.time, time)):
                        return datetime.combine(datetime.today(), val)
                    # Try string parsing
                    return pd.to_datetime(str(val), format='%H:%M', errors='coerce')
                    
                # We need to import datetime and time classes
                from datetime import datetime, time, timedelta

                # Since to_dt might return just a time (if we use pd.to_datetime without date), 
                # let's be careful. pd.to_datetime("09:00:00") returns distinct Timestamp today.
                
                # Robust conversion
                start_dt = None
                if isinstance(start_val, (datetime, time)):
                     start_dt = datetime.combine(datetime.today(), start_val) if isinstance(start_val, time) else start_val
                else:
                     # flexible string parse
                     try:
                         start_dt = pd.to_datetime(str(start_val))
                     except:
                         pass

                end_dt = None
                if isinstance(end_val, (datetime, time)):
                     end_dt = datetime.combine(datetime.today(), end_val) if isinstance(end_val, time) else end_val
                else:
                     try:
                         end_dt = pd.to_datetime(str(end_val))
                     except:
                         pass

                if start_dt and end_dt:
                    # Handle overnight (end < start)
                    if end_dt < start_dt:
                        end_dt += timedelta(days=1)
                    
                    diff = end_dt - start_dt
                    return diff.total_seconds() / 60
                return None
            except Exception as e:
                # print(f"Time Calc Error: {e}")
                return None

        data['수술시간'] = data.apply(calculate_duration, axis=1)

    # Column Check
    required_cols = ['수술시간'] # Age is now optional
    # If Age is missing, we create a dummy or handle it.
    if '연령' not in data.columns:
         data['연령'] = None
         # Try to find '나이' or 'Age'
         for col in ['나이', 'Age', 'AGE']:
             if col in data.columns:
                 data['연령'] = data[col]
                 break

    missing_cols = [col for col in required_cols if col not in data.columns]
    
    if missing_cols:
        # Check if maybe they are in English? 'Age', 'Surgery Time'
        # For now, just raise error
        raise ValueError(f"필수 컬럼이 누락되었습니다: {', '.join(missing_cols)}\n(현재 컬럼: {', '.join(data.columns)})")
    
    
    # 2. Age Binning
    if '연령' in data.columns and data['연령'].notna().any():
        data['연령대'] = data['연령'].apply(bin_age)
    else:
        data['연령대'] = "정보없음"
    
    # 3. Surgery Time Binning
    data['수술시간_범주'] = data['수술시간'].apply(bin_surgery_time)

    # 4. Handle Missing Values
    data.dropna(subset=['수술시간_범주'], inplace=True)
    
    # 5. Explode Nursing Interventions
    if '간호중재_list' in data.columns:
         data['간호중재'] = data['간호중재_list']
         data = data.explode('간호중재')
    elif '간호중재' in data.columns:
        data['간호중재'] = data['간호중재'].astype(str).apply(lambda x: [i.strip() for i in x.split(',') if i.strip()])
        data = data.explode('간호중재')
    else:
         pass
        
    return data

def prepare_transaction_matrix(df, columns_to_include):
    """
    Creates a one-hot encoded transaction matrix for Apriori.
    columns_to_include: list of column names to treat as items (e.g., ['연령대', '수술시간_범주', '간호중재'])
    """
    # Filter only relevant columns
    data_subset = df[columns_to_include].copy()
    
    # Check if columns exist
    existing_cols = [c for c in columns_to_include if c in data_subset.columns]
    if not existing_cols:
        return None
        
    # We need to treat each row as a transaction.
    # Since we want to find associations *between* columns (e.g., Age 60s -> Surgery A),
    # we can prefix the values with the column name to distinguish them.
    # e.g., "Age:60s", "Surgery:Appy"
    
    transactions = []
    for _, row in data_subset.iterrows():
        transaction = []
        for col in existing_cols:
            if pd.notna(row[col]):
                transaction.append(f"{col}:{row[col]}")
        transactions.append(transaction)
        
    return transactions
