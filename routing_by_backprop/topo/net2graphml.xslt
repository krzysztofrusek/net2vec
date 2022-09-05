<?xml version="1.0" encoding="utf-8"?>
<xsl:stylesheet version="1.0" 
	xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
    xmlns:net="http://sndlib.zib.de/network" 
	xmlns:y="http://www.yworks.com/xml/graphml"
>

<xsl:output method="xml" indent="yes"/>

<xsl:template match="net:network/net:networkStructure/net:nodes/net:node">
	<node id="{@id}"  xmlns="http://graphml.graphdrawing.org/xmlns">
		<data key="d0" >
        	<y:ShapeNode>
          		<y:Geometry height="10.0" width="10.0" >
				  	<xsl:attribute name="x"><xsl:value-of select="net:coordinates/net:x"/></xsl:attribute>
					<xsl:attribute name="y"><xsl:value-of select="net:coordinates/net:y"/></xsl:attribute>
				</y:Geometry>
        	</y:ShapeNode>
      	</data>
	</node>
</xsl:template>

<xsl:template match="net:network/net:networkStructure/net:links/net:link">
	<edge  xmlns="http://graphml.graphdrawing.org/xmlns">
		<xsl:attribute name="source"><xsl:value-of select="net:source"/></xsl:attribute>	
		<xsl:attribute name="target"><xsl:value-of select="net:target"/></xsl:attribute>
		
		<data key="d1"><xsl:value-of select="sum(net:additionalModules/net:addModule/net:capacity)"/></data>
		<data key="d2"><xsl:value-of select="sum(net:additionalModules/net:addModule/net:cost)"/></data>
	</edge>
</xsl:template>

<xsl:template match="/">
<graphml xmlns="http://graphml.graphdrawing.org/xmlns" 
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
    xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">
	<key for="node" id="d0" yfiles.type="nodegraphics"/> 
	<key for="edge" id="d1" attr.name="capacity" attr.type="float"/>
	<key for="edge" id="d2" attr.name="cost" attr.type="float"/>
    <graph edgedefault="undirected">
  	    <xsl:apply-templates select="net:network/net:networkStructure/net:nodes/net:node" />
	    <xsl:apply-templates select="net:network/net:networkStructure/net:links/net:link"/>
    </graph>
</graphml>

</xsl:template>


</xsl:stylesheet>
