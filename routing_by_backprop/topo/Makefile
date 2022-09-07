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


	
	