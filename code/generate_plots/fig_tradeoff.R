
library(ggplot2)
library(dplyr)
library(cowplot)

## NOTE FOR THE ICONS USAGE
# Some icons are with a different ratio than others, please check the plot and adjust the xmin and xmax of the annotation_custom accordingly 

#select the column
selected <- "gpt4_acme" # or column name
base_data <- read.csv("model_preferences_by_model.csv")



ederly <- readPNG("icons/icons_tradeoff/ederly.png")
female <-  readPNG("icons/icons_tradeoff/female.png")
fewer <-  readPNG("icons/icons_tradeoff/fewer.png")
fit <-  readPNG("icons/icons_tradeoff/fit.png")
higher_status <-  readPNG("icons/icons_tradeoff/higher_status.png")
humans <-  readPNG("icons/icons_tradeoff/humans.png")
large <- readPNG("icons/icons_tradeoff/large.png")
lower_status <- readPNG("icons/icons_tradeoff/lower_status.png")
male <- readPNG("icons/icons_tradeoff/male.png")
more_character <- readPNG("icons/icons_tradeoff/more_character.png")
pet <- readPNG("icons/icons_tradeoff/pet.png")
young <- readPNG("icons/icons_tradeoff/young.png")
icons_left <- list(pet, ederly, large, male, lower_status, fewer)
icons_right <- list(humans, young, fit, female, higher_status, more_character)

human_data <- read.csv("model_preferences_by_model.csv")
human_data <- human_data$human
human_data <- (human_data - (100 - human_data))/100

data <- base_data[, c("criterion", selected)]
data[,2] <- (data[,2] - (100 -data[,2]))/100
data$human <- human_data
names(data) <- c("criterion", "lang", "human")

label_left <- c(
  expression(paste(bold("Species"),"                                                 Sparing pets")),
  expression(paste(bold("Age"),"                                              Sparing the elderly")),
  expression(paste(bold("Fitness"),"                                          Sparing the large ")),
  expression(paste(bold("Gender"),"                                               Sparing males")),
  expression(paste(bold("Social Status"),"                            Sparing lower status")),
  expression(paste(bold("No. characters"),"                   Sparing fewer characters"))
)

label_right <- c( 
  "  Sparing humans",
  "  Sparing the young",
  "  Sparing the fit",
  "  Sparing females",
  "  Sparing higher status",
  "  Sparing more characters"
)

label_left <- label_left[order(data$lang, decreasing = TRUE)]
label_right <- label_right[order(data$lang, decreasing = TRUE)]
icons_left <- icons_left[order(data$lang, decreasing = TRUE)]
icons_right <- icons_right[order(data$lang, decreasing = TRUE)]
data<- data[order(data$lang,  decreasing = TRUE),]
data$criterion1 <- c("a", "b", "c", "d","e", "f")
names <- c("1","2","3","4","5","6")
ggplot() + 
  geom_col(aes(y=data$lang, x=as.numeric(as.factor(data$criterion1))), fill="#00557b", width = .3, alpha = 1) +
  geom_point(aes(y=data$human, x=as.numeric(as.factor(data$criterion1)) ),  size=8, col="#fadc02")+
  theme_bw()+
  scale_x_continuous(
    breaks = 1:6,
    labels = label_left,
    sec.axis = sec_axis(~. ,breaks = 1:6, labels= label_right)
  )+
  ylab(expression(paste("\n ",Delta," Pr")))+
  scale_y_continuous(limits = c(-0.05, 1.05) )+
  theme(
    legend.position = "none",
    axis.text.y = element_text(hjust=0,color="black"),
    axis.title.x = element_text(size=29),
    axis.title.y = element_blank(),
    axis.text= element_text(size=25),
    aspect.ratio=1/2,
    plot.title = element_text(size=30, hjust = 0.5),
    panel.grid.major.y = element_line(linewidth = 2),
    panel.grid.minor.y = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    panel.border  = element_blank(),
    axis.ticks.y = element_blank()
  )+
  coord_flip()+
  ggtitle("Preference in favor of the choice on the right side")+
  annotation_custom(rasterGrob(icons_left[[1]], interpolate=FALSE),xmin=0.6, xmax=1.4, ymin=-0.46, ymax=0.3) +
  annotation_custom(rasterGrob(icons_left[[2]], interpolate=FALSE), xmin=1.6, xmax=2.4, ymin=-0.46, ymax=0.3) +
  annotation_custom(rasterGrob(icons_left[[3]], interpolate=FALSE), xmin=2.6, xmax=3.4, ymin=-0.46, ymax=0.3) +
  annotation_custom(rasterGrob(icons_left[[4]], interpolate=FALSE), xmin=3.6, xmax=4.4, ymin=-0.46, ymax=0.3) +
  annotation_custom(rasterGrob(icons_left[[5]], interpolate=FALSE), xmin=4.6, xmax=5.4, ymin=-0.46, ymax=0.3) +
  annotation_custom(rasterGrob(icons_left[[6]], interpolate=FALSE), xmin=5.6, xmax=6.4, ymin=-0.46, ymax=0.3) +
  
  annotation_custom(rasterGrob(icons_right[[1]], interpolate=FALSE),        xmin=0.6, xmax=1.3,     ymin=0.83, ymax=1.3) +
  annotation_custom(rasterGrob(icons_right[[2]], interpolate=FALSE),        xmin=1.6, xmax=2.3,    ymin=0.83, ymax=1.3) +
  annotation_custom(rasterGrob(icons_right[[3]], interpolate=FALSE),        xmin=2.6, xmax=3.4,     ymin=0.85, ymax=1.3) +
  annotation_custom(rasterGrob(icons_right[[4]], interpolate=FALSE),        xmin=3.6, xmax=4.4,    ymin=0.85, ymax=1.3) +
  annotation_custom(rasterGrob(icons_right[[5]], interpolate=FALSE),        xmin=4.6, xmax=5.4,     ymin=0.85, ymax=1.3) +
  annotation_custom(rasterGrob(icons_right[[6]], interpolate=FALSE),        xmin=5.6, xmax=6.4,     ymin=0.85, ymax=1.3) 
ggsave("PLOT/fig_tradeoff.pdf", device=pdf, width = 27, height = 10)


