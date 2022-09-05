import networkx as nx
import glob
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

with PdfPages('sndlib.pdf') as pdf:
    for gml in glob.glob('sndlib-networks-xml/*.graphml'):
        G = nx.read_graphml(gml)
        # G=nx.convert_node_labels_to_integers(G)
        x = nx.get_node_attributes(G, 'x')
        y = nx.get_node_attributes(G, 'y')

        pos = {k: (float(v), float(y[k])) for k, v in x.items()}

        plt.figure(figsize=(11.69, 8.27))
        nx.draw_networkx(G, pos=pos)
        plt.title(f'{gml} ({len(G)})')
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()

        if len(G) <= 40:
            print(gml)

