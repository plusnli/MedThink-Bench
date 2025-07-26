import re
import pandas as pd
import xlsxwriter


def parse_output(output):
    # Find pattern like "[A. option1]" or "[B. option2]"
    pattern0 = r'\[\s*[A-J]\.\s*[A-Za-z0-9\s\+]+\s*\]'
    matches = re.findall(pattern0, output)
    if matches:
        if matches[-1].strip()[0] == "[":
            return matches[-1].strip()[1:-1].strip()
        else:
            return matches[-1].strip()

    # Try to match pattern like "B. Atropine" or "D. IMF + ID"
    pattern1 = r'[A-J]\.\s*[A-Za-z0-9\s\+]+'
    matches = re.findall(pattern1, output)
    if matches:
        return matches[-1].strip()
    
    # If not found, try to match pattern surrounded by **
    pattern2 = r'\*\*([A-J]\.\s*[A-Za-z0-9\s\+]+)\*\*'
    matches = re.findall(pattern2, output)
    if matches:
        return matches[-1].strip()
    
    # Try to match pattern like "A: xxx" or "B: xxx"
    pattern3 = r'[A-J]:\s*[A-Za-z0-9\s\+]+'
    matches = re.findall(pattern3, output)
    if matches:
        return matches[-1].strip()
    
    # Try to match pattern like "(A) xxx" or "(B) xxx"
    pattern4 = r'\(([A-J])\)\s*[A-Za-z0-9\s\+]+'
    matches = re.findall(pattern4, output)
    if matches:
        return matches[-1].strip()

    # Try to match pattern like "(A). xxx" or "(B). xxx"
    pattern5 = r'\(([A-J])\)\.\s*[A-Za-z0-9\s\+]+'
    matches = re.findall(pattern5, output)
    if matches:
        return matches[-1].strip()

    # Try to match single uppercase letter (No matter if it is surrounded by ** or not)
    pattern6 = r'(?:\*\*)?([A-J])(?:\*\*)?'
    matches = re.findall(pattern6, output)
    if matches:
        return matches[-1].strip()

    # If not found, return empty string
    return ""


def parse_output_old(output):
    # Find pattern like "B. Atropine" or "D. IMF + ID" or single uppercase letter
    # Try to match pattern like "B. Atropine" or "D. IMF + ID"
    pattern1 = r'[A-H]\.\s*[A-Za-z0-9\s\+]+' 
    matches = re.findall(pattern1, output)
    if matches:
        return matches[-1].strip()
    
    # If not found, try to match pattern surrounded by **
    pattern2 = r'\*\*([A-H]\.\s*[A-Za-z0-9\s\+]+)\*\*'
    matches = re.findall(pattern2, output)
    if matches:
        return matches[-1].strip()
    
    # Try to match single uppercase letter
    pattern3 = r'\*\*([A-H])\*\*'
    matches = re.findall(pattern3, output)
    if matches:
        return matches[-1].strip()
    
    # If not found, return empty string
    return ""


def beautify_write_excel(df_results, results_path):
    # Beautify the Excel file
    with pd.ExcelWriter(results_path, engine='xlsxwriter') as writer:
        # 1. write the basic results
        df_results.to_excel(writer, sheet_name='Results', index=False)

        # 2. get the workbook and worksheet
        workbook  = writer.book
        worksheet = writer.sheets['Results']

        # 3. header format: bold, text wrap, vertical center, background color, border
        header_format = workbook.add_format({
            'bold':      True,
            'text_wrap': True,
            'valign':    'center',
            'fg_color':  '#D7E4BC',
            'border':    1
        })
        for col_num, col_name in enumerate(df_results.columns):
            worksheet.write(0, col_num, col_name, header_format)

        # 4. set the column width to one eighth of the string length
        for i, col in enumerate(df_results.columns):
            max_len = df_results[col].astype(str).map(len).max()
            max_len = max(max_len, len(col))
            col_width = max_len // 8  # set the column width to one eighth of the string length
            col_width = max(col_width, 5)  # ensure the minimum width is 5
            worksheet.set_column(i, i, col_width)

        # 5. freeze the first row + auto filter
        worksheet.freeze_panes(1, 0)
        worksheet.autofilter(0, 0, df_results.shape[0], df_results.shape[1] - 1)

        # 6. color the numeric columns
        for col in df_results.select_dtypes(include=['number']).columns:
            idx = df_results.columns.get_loc(col)
            worksheet.conditional_format(1, idx, df_results.shape[0], idx, {
                'type':      '3_color_scale',
                'min_color': '#63BE7B',
                'mid_color': '#FFEB84',
                'max_color': '#F8696B'
            })
    print(f"The beautified Excel file has been saved to: {results_path}")
    