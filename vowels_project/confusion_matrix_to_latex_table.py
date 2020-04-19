import numpy as np


def to_latex_table(confusion_matrix, data_labels, filename, table_caption='Confusion matrix', table_label='tab:',
                   true_label='True value', predicted_label='Predicted value'):
    n = len(data_labels)
    f = open(filename, 'w')
    f.write('\\begin{table}[H]\n')
    f.write('\\centering\n')
    f.write('\\caption{' + table_caption + '}\n')
    f.write('\\label{' + table_label + '}\n')
    f.write('\\begin{tabular}{' + '|l' * (2 + n) + '|' + '} \\hline\n')
    start_of_line = ' ' * (25 + len(predicted_label) + len(str(n)))
    f.write(start_of_line + '&    ' + '& \\multicolumn{' + str(n) + '}{c|}{' + true_label + '} \\\\ \\hline\n')
    f.write(start_of_line + '&    ')
    for label in data_labels:
        f.write('& ' + label + ' ')
    f.write('\\\\ \\hline\n')
    f.write('\\parbox[t]{2mm}{\multirow{' + str(n) + '}{*}{\\rotatebox[origin=c]{90}{' + predicted_label + '}}}\n')
    for i in range(len(data_labels)):
        f.write(start_of_line)
        f.write('& ' + data_labels[i] + ' ')
        for j in range(len(data_labels)):
            spacing = ' ' if confusion_matrix[i][j] < 10 else ''
            f.write('& ' + spacing + str(confusion_matrix[i][j]) + ' ')
        if i == len(data_labels) - 1:
            f.write('\\\\ \\hline\n')
        else:
            f.write('\\\\ \\cline{2-' + str(n+2) + '}\n')
    f.write('\\end{tabular}\n')
    f.write('\\end{table}\n')
