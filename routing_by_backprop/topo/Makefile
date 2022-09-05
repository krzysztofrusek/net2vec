PLG_PATH = plgkrusek@prometheus.cyfronet.pl:/net/scratch/people/plgkrusek/routing_by_backprop
XML_DIR = sndlib-networks-xml

sndlib-networks-xml.tgz:
	curl -o $@ http://sndlib.zib.de/download/sndlib-networks-xml.tgz
	
untar: sndlib-networks-xml.tgz
	tar -xvkf sndlib-networks-xml.tgz
	
$(XML_DIR)/%.xml:untar
	

$(XML_DIR)/%.graphml: $(XML_DIR)/%.xml
	xsltproc -o $@ net2graphml.xslt $<

.PHONY: clean
	
clean:
	rm -rf  sndlib-networks-xml.tgz sndlib-networks-xml
	
all:$(patsubst %.xml,%.graphml,$(wildcard $(XML_DIR)/*)) sndlib-networks-xml.tgz untar

pdf:all
	python3 book.py
test:all
	python -m unittest test.py
	
sync: all
	rsync -avx --inplace --delete \
	  sndlib-networks-xml/abilene.graphml \
	  sndlib-networks-xml/pdh.graphml \
	  sndlib-networks-xml/giul39.graphml \
	  sndlib-networks-xml/nobel-germany.graphml \
	  sndlib-networks-xml/france.graphml \
	  sndlib-networks-xml/janos-us.graphml \
	  sndlib-networks-xml/ta1.graphml \
	  sndlib-networks-xml/di-yuan.graphml \
	  sndlib-networks-xml/geant.graphml \
	  sndlib-networks-xml/india35.graphml \
	  sndlib-networks-xml/cost266.graphml \
	  sndlib-networks-xml/pioro40.graphml \
	  sndlib-networks-xml/dfn-gwin.graphml \
	  sndlib-networks-xml/newyork.graphml \
	  sndlib-networks-xml/atlanta.graphml \
	  sndlib-networks-xml/sun.graphml \
	  sndlib-networks-xml/dfn-bwin.graphml \
	  sndlib-networks-xml/polska.graphml \
	  sndlib-networks-xml/nobel-eu.graphml \
	  sndlib-networks-xml/nobel-us.graphml \
	  sndlib-networks-xml/norway.graphml \
	  sndlib-networks-xml/janos-us-ca.graphml \
	  "$(PLG_PATH)/net"
	

	
	