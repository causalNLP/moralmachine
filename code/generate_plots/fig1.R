
library(ggplot2)
library(dplyr)
library(cowplot)

data <- read.csv("model_preferences_by_model.csv")
data[,2] <- (data[,2] - (100-data[,2])  )/100
data[,3] <- (data[,3]  - (100-data[,3]) )/100
data<- data[order(data$gpt4_acme,  decreasing = TRUE),]
data$criterion1 <- c("a", "b", "c", "d","e", "f")
names <- c("1","2","3","4","5","6")
ggplot() + 
  #geom_vline(xintercept=as.numeric(as.factor(data$criterion1)), size=0.3, linetype="dotted", col="darkgrey" )+
  #geom_hline(yintercept=0, col="darkgrey")+
  #geom_hline(yintercept=1, col="darkgrey")+

  geom_col(aes(y=data$gpt4_acme, x=as.numeric(as.factor(data$criterion1))), fill="#0077ad", width = .3, alpha = 0.7) +
  geom_point(aes(y=data$gpt4_acme, x=as.numeric(as.factor(data$criterion1))),size = 11, shape = 21, fill = "white", stroke = 2, col="#0077ad")+
  geom_point(aes(y=data$human, x=as.numeric(as.factor(data$criterion1)) ),  size=8, col="#fadc02")+
  theme_bw()+
  scale_x_continuous(
    breaks = 1:6,
    labels = c(
      expression(paste(bold("Species"),"                                                  Sparing pets          ")),
      expression(paste(bold("No. characters"),"                  Sparing fewer characters          ")),
      expression(paste(bold("Age"),"                                               Sparing the elderly          ")),
      expression(paste(bold("Social Status"),"                            Sparing lower status          ")),
      expression(paste(bold("Fitness"),"                                            Sparing the large          ")),
      expression(paste(bold("Gender"),"                                                Sparing males          "))
              ),
    sec.axis = sec_axis(~. ,breaks = 1:6, labels= c( 
      "           Sparing humans",
       "           Sparing more characters",
       "           Sparing the young",
       "           Sparing higher status",
       "           Sparing the fit",
      "           Sparing females"
    ))
  )+
  ylab(expression(paste("\n ",Delta," Pr")))+
  #scale_y_continuous(breaks = c(10,20,30,40, 50,60), labels = c("10","2","3","4","3","4"))+
  theme(
        legend.position = "none",
        axis.text.y = element_text(hjust=0,color="black"),
        axis.title.x = element_text(size=29),
        axis.title.y = element_blank(),
        axis.text= element_text(size=25),
        aspect.ratio=1/2,
        plot.title = element_text(size=22),
        panel.border = element_rect(colour = "black", fill=NA, size=1.5),
        #panel.grid.major.y = element_blank(),
        panel.grid.minor.y = element_blank(),
        )+
  ggtitle("Preference in favor of the choice on the right side")+
  coord_flip()

ggsave("fig_bar_pref2a.pdf", device=cairo_pdf, width = 27, height = 10, dpi=800)


