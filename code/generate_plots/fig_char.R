library(ggplot2)
library(png)
library(gridGraphics)

plotdata <- read.csv("human_char_pref_scores.csv")
plotdata <- plotdata[,c(2,5)] #select the column
colnames(plotdata) <- c("CharacterType", "Estimates")
order_ <- order(plotdata$Estimates, decreasing = FALSE)
plotdata <- plotdata[order_, ]

yticks <- seq(-0.8,0.8,0.2)


stroller <- readPNG("icons/stroller.png")
girl <- readPNG("icons/girl.png")
boy <- readPNG("icons/boy.png")
pregnant <- readPNG("icons/pregnant.png")
maledoctor <- readPNG("icons/maledoctor.png")
femaledoctor <- readPNG("icons/femaledoctor.png")
femaleathlete <- readPNG("icons/femaleathlete.png")
femaleexecutive <- readPNG("icons/femaleexecutive.png")
maleathlete <- readPNG("icons/maleathlete.png")
maleexecutive <- readPNG("icons/maleexecutive.png")
largewoman <- readPNG("icons/femalelarge.png")
largeman <- readPNG("icons/malelarge.png")
homeless <- readPNG("icons/homeless.png")
oldman <- readPNG("icons/oldman.png")
oldwoman <- readPNG("icons/oldwoman.png")
dog <- readPNG("icons/dog.png")
criminal <- readPNG("icons/criminal.png")
cat<- readPNG("icons/cat.png")
icons <- list(boy, cat, criminal, dog, oldman, oldwoman, femaleathlete, femaledoctor, femaleexecutive, girl, homeless, largeman, largewoman, maleathlete, maledoctor, maleexecutive, pregnant, stroller )
icons <- icons[order_]
g1 <- rasterGrob(stroller, interpolate=FALSE)
ggirl <-  rasterGrob(girl, interpolate=FALSE)



ggplot(plotdata,aes(reorder(CharacterType, Estimates), Estimates,color=Estimates>0,fill=Estimates>0))+
  geom_col(width = 0.8, alpha=1)+
  theme_bw() + coord_flip()+
  scale_y_continuous(limits = c(-0.65,0.5),breaks=yticks,labels=sapply(yticks,function(z) return(ifelse(z<0,paste0("-  ",as.character(-z)),ifelse(z>0,paste0("+",as.character(z)),"Person")))))+
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
  annotation_custom(rasterGrob(icons[[18]],  interpolate=FALSE), xmin=17.5, xmax=18.5, ymin=-1.15, ymax=-0.22) +
  annotation_custom(rasterGrob(icons[[17]],  interpolate=FALSE), xmin=16.5, xmax=17.5, ymin=-1.15, ymax=-0.22) +
  annotation_custom(rasterGrob(icons[[16]],  interpolate=FALSE), xmin=15.5, xmax=16.5, ymin=-1.15, ymax=-0.22) +
  annotation_custom(rasterGrob(icons[[15]],  interpolate=FALSE), xmin=14.5, xmax=15.5, ymin=-1.15, ymax=-0.22) +
  annotation_custom(rasterGrob(icons[[14]],  interpolate=FALSE), xmin=13.5, xmax=14.5, ymin=-1.15, ymax=-0.22) +
  annotation_custom(rasterGrob(icons[[13]],  interpolate=FALSE), xmin=12.5, xmax=13.5, ymin=-1.15, ymax=-0.22) +
  annotation_custom(rasterGrob(icons[[12]],  interpolate=FALSE), xmin=11.5, xmax=12.5, ymin=-1.15, ymax=-0.22) +
  annotation_custom(rasterGrob(icons[[11]],  interpolate=FALSE), xmin=10.5, xmax=11.5, ymin=-1.15, ymax=-0.22) +
  annotation_custom(rasterGrob(icons[[10]],  interpolate=FALSE), xmin=9.5, xmax=10.5, ymin=-1.15, ymax=-0.22) +
  annotation_custom(rasterGrob(icons[[9]], interpolate=FALSE), xmin=8.5, xmax=9.5, ymin=-1.15, ymax=-0.22) +
  annotation_custom(rasterGrob(icons[[8]], interpolate=FALSE), xmin=7.5, xmax=8.5, ymin=-1.15, ymax=-0.22) +
  annotation_custom(rasterGrob(icons[[7]], interpolate=FALSE), xmin=6.5, xmax=7.5, ymin=-1.15, ymax=-0.22) +
  annotation_custom(rasterGrob(icons[[6]], interpolate=FALSE), xmin=5.5, xmax=6.5, ymin=-1.15, ymax=-0.22) +
  annotation_custom(rasterGrob(icons[[5]], interpolate=FALSE), xmin=4.5, xmax=5.5, ymin=-1.15, ymax=-0.22)+
  annotation_custom(rasterGrob(icons[[4]], interpolate=FALSE), xmin=3.5, xmax=4.5, ymin=-1.15, ymax=-0.22) +
  annotation_custom(rasterGrob(icons[[3]], interpolate=FALSE), xmin=2.5, xmax=3.5, ymin=-1.15, ymax=-0.22) +
  annotation_custom(rasterGrob(icons[[2]], interpolate=FALSE), xmin=1.5, xmax=2.5, ymin=-1.15, ymax=-0.22) +
  annotation_custom(rasterGrob(icons[[1]], interpolate=FALSE), xmin=0.5, xmax=1.5, ymin=-1.15, ymax=-0.22) 
ggsave("PLOT/fig_char.pdf", device="pdf", width = 28, height = 17.5)

