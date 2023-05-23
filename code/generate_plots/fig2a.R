
library(ggplot2)
library(dplyr)
library(cowplot)

data <- read.csv("model_preferences_all.csv")
#data[,2] <- (data[,2] - (100-data[,2])  )/100
#data[,11] <- (data[,11]  - (100-data[,11]) )/100
data[,2] <- (data[,2]  )/100
data[,11] <- (data[,11] )/100
data<- data[order(data$gpt4,  decreasing = TRUE),]
data$criterion <- c("a", "b", "c", "d","e", "f")
names <- c("1","2","3","4","5","6")
ggplot() + 
  #geom_vline(xintercept=as.numeric(as.factor(data$criterion1)), size=0.3, linetype="dotted", col="darkgrey" )+
  #geom_hline(yintercept=0, col="darkgrey")+
  #geom_hline(yintercept=1, col="darkgrey")+

  geom_col(aes(y=data$gpt4, x=as.numeric(as.factor(data$criterion))), fill="#0077ad", width = .3, alpha = 0.7) +
  geom_point(aes(y=data$gpt4, x=as.numeric(as.factor(data$criterion))),size = 11, shape = 21, fill = "white", stroke = 2, col="#0077ad")+
  geom_point(aes(y=data$human, x=as.numeric(as.factor(data$criterion)) ),  size=15, col="#fadc02")+
  theme_bw()+
  scale_x_continuous(
    breaks = 1:6,
    labels = c(
      expression(paste(bold("           Species"),"                                       Sparing pets  ")),
      expression(paste(bold("No. characters"),"                   Sparing fewer characters  ")),
      expression(paste(bold("                  Age"),"                              Sparing the elderly  ")),
      expression(paste(bold("   Social Status"),"                           Sparing lower status  ")),
      expression(paste(bold("             Fitness"),"                               Sparing the large  ")),
      expression(paste(bold("             Gender"),"                                    Sparing males  "))
              ),
    sec.axis = sec_axis(~. ,breaks = 1:6, labels= c( 
      "  Sparing humans",
       "  Sparing more characters",
       "  Sparing the young",
       "  Sparing higher status",
       "  Sparing the fit",
      "  Sparing females"
    ))
  )+

  ylab(expression(paste("\n Pr")))+
#  annotate("text", x = 6, y = 0.5, label = "Human Study Result", color = "#A96904", size = 8)+
#  annotate("text", x = 6, y = 0.125, label = "LLM's Preference", color = "#0077ad", size = 8)+
  annotate("text", x = 6, y = 0.82, label = "Human Study Result", color = "#A96904", size = 8)+
  annotate("text", x = 6, y = 0.125, label = "LLM's Preference", color = "white", size = 8)+
  scale_y_continuous(breaks=c(0,0.25,0.5,0.75,1),
                   labels=c("100% Favoring the Left", "75-25", "50-50","25-75","100% Favoring the Right"))+
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
  #ggtitle("Preference in favor of the choice on the right side")+
  coord_flip()

ggsave("PLOT/fig_tradeoff_normal.pdf", device=cairo_pdf, width = 27, height = 10)

#############################################
plotdata <- data
yticks <- seq(-0.2,0.8,0.2)

ggplot(plotdata,aes(x=criterion, y=gpt4))+
  geom_hline(yintercept=0, color="black", linewidth=0.7)+
  geom_col(width = .5, fill = "#0077ad", alpha = 0.5)+
# geom_col(data = plotdata.util, width = .5, fill = "#0077ad", alpha = .7,aes(y=max(Estimates)))+
  #geom_point(data = plotdata.util, color = "#49c6ff" , size = 5.5, shape = 21, fill = "white", stroke = 2)+  
  geom_point(size = 5.5, shape = 21, color = "#0077ad", fill = "white", stroke = 2)+
  #geom_text(data=plotdata.util, aes(label=as.character(c(1:4)),y=Estimates),size=3,color="#37a2ef")+
  coord_flip()+
  labs(title="Preference in favor of the choice on the right side")+
  xlab("")+
  ylab(expression(paste("\n "," Pr")))+
  scale_y_continuous(limits = c(-0.2,0.9),breaks=yticks,labels=sapply(yticks,function(z) return(ifelse(z<0,"",ifelse(z>0,paste0("+",as.character(z)),"no change")))))+
  theme_bw()+


  theme(text=element_text(size=20),
        legend.position="right",
        aspect.ratio=1/2, 
        axis.title.x = element_text(size=20),
        axis.text.y = element_text(hjust=0,color="black"),
        legend.text=element_text(size=8),
        panel.border = element_rect(colour = "black", fill=NA, size=1.5))


