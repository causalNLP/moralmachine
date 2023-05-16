library(ggplot2)
library(png)
library(gridGraphics)
plotdata <- read.csv("human_char_pref_scores_updated.csv")
plotdata <- plotdata[,c(2,5)]
colnames(plotdata) <- c("CharacterType", "Estimates")
plotdata <- plotdata[order(plotdata$Estimates, decreasing = FALSE), ]

yticks <- seq(-0.8,0.8,0.2)


stroller <- readPNG("PLOT/icons/stroller.png")
girl <- readPNG("PLOT/icons/girl.png")
boy <- readPNG("PLOT/icons/boy.png")
pregnant <- readPNG("PLOT/icons/pregnant.png")
maledoctor <- readPNG("PLOT/icons/maledoctor.png")
femaledoctor <- readPNG("PLOT/icons/femaledoctor.png")
femaleathlete <- readPNG("PLOT/icons/femaleathlete.png")
femaleexecutive <- readPNG("PLOT/icons/femaleexecutive.png")
maleathlete <- readPNG("PLOT/icons/maleathlete.png")
maleexecutive <- readPNG("PLOT/icons/maleexecutive.png")
largewoman <- readPNG("PLOT/icons/femalelarge.png")
largeman <- readPNG("PLOT/icons/malelarge.png")
homeless <- readPNG("PLOT/icons/homeless.png")
oldman <- readPNG("PLOT/icons/oldman.png")
oldwoman <- readPNG("PLOT/icons/oldwoman.png")
dog <- readPNG("PLOT/icons/dog.png")
criminal <- readPNG("PLOT/icons/criminal.png")
cat<- readPNG("PLOT/icons/cat.png")


g1 <- rasterGrob(stroller, interpolate=FALSE)
ggirl <-  rasterGrob(girl, interpolate=FALSE)



ggplot(plotdata,aes(reorder(CharacterType, Estimates), Estimates,color=Estimates>0,fill=Estimates>0))+
  geom_col(width = 0.8, alpha=1)+
  theme_bw() + coord_flip()+
  scale_y_continuous(limits = c(-0.65,0.5),breaks=yticks,labels=sapply(yticks,function(z) return(ifelse(z<0,paste0("-  ",as.character(-z)),ifelse(z>0,paste0("+",as.character(z)),"no change")))))+
  scale_color_manual(values = c("#B74F6F", "#0077AD"))+
  scale_fill_manual(values = c("#B74F6F", "#0077AD"))+
  xlab("Characters")+
  ylab("Preference Score")+
  theme(
    
    legend.position = "none",
    panel.grid.major.y = element_line(linewidth = 2),
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    panel.border  = element_blank(),
    axis.ticks.y = element_blank(),
    axis.text.y = element_text(size=45, color = "black"),
    axis.text.x = element_text(size=45, color = "black"),
    axis.title = element_text(size=45),
    plot.title = element_text(size=50, hjust=0.5)
  )+ 
  ggtitle("Preference in Favor of Sparing Characters")+
  annotation_custom(rasterGrob(stroller, interpolate=FALSE), xmin=17.5, xmax=18.5, ymin=-1.15, ymax=-0.22) +
  annotation_custom(rasterGrob(girl, interpolate=FALSE), xmin=16.5, xmax=17.5, ymin=-1.15, ymax=-0.22) +
  annotation_custom(rasterGrob(boy, interpolate=FALSE), xmin=15.5, xmax=16.5, ymin=-1.15, ymax=-0.22) +
  annotation_custom(rasterGrob(pregnant, interpolate=FALSE), xmin=14.5, xmax=15.5, ymin=-1.15, ymax=-0.22) +
  annotation_custom(rasterGrob(maledoctor, interpolate=FALSE), xmin=13.5, xmax=14.5, ymin=-1.15, ymax=-0.22) +
  annotation_custom(rasterGrob(femaledoctor, interpolate=FALSE), xmin=12.5, xmax=13.5, ymin=-1.15, ymax=-0.22) +
  annotation_custom(rasterGrob(femaleathlete, interpolate=FALSE), xmin=11.5, xmax=12.5, ymin=-1.15, ymax=-0.22) +
  annotation_custom(rasterGrob(femaleexecutive, interpolate=FALSE), xmin=10.5, xmax=11.5, ymin=-1.15, ymax=-0.22) +
  annotation_custom(rasterGrob(maleathlete, interpolate=FALSE), xmin=9.5, xmax=10.5, ymin=-1.15, ymax=-0.22) +
  annotation_custom(rasterGrob(maleexecutive, interpolate=FALSE), xmin=8.5, xmax=9.5, ymin=-1.15, ymax=-0.22) +
  annotation_custom(rasterGrob(largewoman, interpolate=FALSE), xmin=7.5, xmax=8.5, ymin=-1.15, ymax=-0.22) +
  annotation_custom(rasterGrob(largeman, interpolate=FALSE), xmin=6.5, xmax=7.5, ymin=-1.15, ymax=-0.22) +
  annotation_custom(rasterGrob(homeless, interpolate=FALSE), xmin=5.5, xmax=6.5, ymin=-1.15, ymax=-0.22) +
  annotation_custom(rasterGrob(oldman, interpolate=FALSE), xmin=4.5, xmax=5.5, ymin=-1.15, ymax=-0.22)+
  annotation_custom(rasterGrob(oldwoman, interpolate=FALSE), xmin=3.5, xmax=4.5, ymin=-1.15, ymax=-0.22) +
  annotation_custom(rasterGrob(dog, interpolate=FALSE), xmin=2.5, xmax=3.5, ymin=-1.15, ymax=-0.22) +
  annotation_custom(rasterGrob(criminal, interpolate=FALSE), xmin=1.5, xmax=2.5, ymin=-1.15, ymax=-0.22) +
  annotation_custom(rasterGrob(cat, interpolate=FALSE), xmin=0.5, xmax=1.5, ymin=-1.15, ymax=-0.22) 
ggsave("PLOT/fig_char.pdf", device="pdf", width = 28, height = 17.5)
           
