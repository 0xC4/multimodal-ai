other_fusion_name <- function(x){
  # Simple function that returns the opposite fusion type i.e. late --> early and early --> late
  
  if(x == "early"){
    return("late")
  }
  return("early")
}

find_curve <- function(curve_list, location, model, fusion){
  # Retrieves an ROC curve from a list of ROC curves by its attributes
  #   curve_list: list of ROC curves
  #   location:   dataset location (interal / external)
  #   fusion:     fusion type (early / late) 
  
  Filter(function(x) x[["location"]] == location & x[["model"]] == model & x[["fusion"]] == fusion, curve_list)[[1]]
}

pretty_model_name <- function(model, fusion){
  # Returns a stylized model descriptor for figure legends
  #   model:  original modal name as assigned in the code
  #   fusion: fusion type (early / late)
  
  f = "unknown"
  if(fusion == "single"){
    f = "Unimodal"
  }
  else if(fusion == "early"){
    f = "Early fusion"
  }
  else if(fusion == "late"){
    f = "Multimodal"
  }
  
  if(model == "ai_with_clinical") {
    m = "DL susp.+clin.+vol."
  }
  else if(model == "dl_clinical"){
    m = "DL susp.+clin."
  }
  else if(model == "dl_volumetric"){
    m = "DL susp.+vol."
  }
  else if(model == "baseline"){
    m = "DL"
  }
  else if(model == "baseline_multi"){
    m = "DL susp."
  }
  else if(model == "clinical"){
    m = "Clinical"
  }
  else if(model == "pirads"){
    m = "pirads"
  }
  
  prettyname = paste0(f, " ", m)
  return(prettyname)
}

# Read all result files
data = data.frame()
for (filepath in Sys.glob("/path/to/files/prediction_results/*.csv")){
  print(filepath)
  data = rbind(data, read.csv2(filepath))
}

# Convert each column to its appropriate typing
data$pred.num = as.numeric(data$pred)
data$label.num = as.numeric(data$label)
data$dataset.fac = as.factor(data$dataset)
data$model.fac = as.factor(data$model)

# Create exceptions for unimodal baselines and PI-RADS scores (they don't have early/late fusion)
data$fusion[data$model %in% c("clinical", "baseline", "baseline_multi", "pirads")] = "single"
data$fusion.fac = as.factor(data$fusion)

# Calculate ROC curves for each predictor, and store it in a list
library(pROC)
roc_curves = list()

for (location_ in c(unique(data$dataset))){
  roc_curves[[location_]] = list()
  
  print(location_)
  d1 = subset(data, dataset==location_)
  
  for (fusion_ in unique(d1$fusion)){
    print(fusion_)
    d2 = subset(d1, fusion==fusion_)
    
    for (model_ in unique(d2$model)){
      print(model_)
      d3 = subset(d2, model==model_)
      curve = pROC::roc(d3$label.num, d3$pred.num)
      auc_ = as.character(round(curve$auc, 3))
      pname = pretty_model_name(model_, fusion_)
      
      roc_curves[[location_]][[pname]] = curve
      
      # Also add additional attributes to the curve to make it findable
      roc_curves[[location_]][[pname]]$model = model_
      roc_curves[[location_]][[pname]]$fusion = fusion_
      roc_curves[[location_]][[pname]]$location = location_
    }
  }
}

# Retrieve specific curves from the list of all curves
internal_curves = roc_curves$internal
internal_ai_curves = Filter(function(x) x[["fusion"]] != "single", internal_curves)
internal_baseline = Filter(function(x) x[["model"]] == "baseline", internal_curves)[[1]]
internal_clinical = Filter(function(x) x[["model"]] == "clinical", internal_curves)[[1]]
internal_pirads = Filter(function(x) x[["model"]] == "pirads", internal_curves)[[1]]
internal_all_curves_except_pirads = Filter(function(x) !(x[["model"]] %in% c("pirads", "baseline_multi")), internal_curves)
internal_early_all_curves_except_pirads = Filter(function(x) x[["fusion"]] %in% c("early", "single"), internal_all_curves_except_pirads)
internal_late_all_curves_except_pirads = Filter(function(x) x[["fusion"]] %in% c("late", "single"), internal_all_curves_except_pirads)
internal_baseline_multi_lesion = Filter(function(x) x[["model"]] == "baseline_multi", internal_curves)[[1]]

external_curves = roc_curves$external
external_ai_curves = Filter(function(x) x[["fusion"]] != "single", external_curves)
external_baseline = Filter(function(x) x[["model"]] == "baseline", external_curves)[[1]]
external_clinical = Filter(function(x) x[["model"]] == "clinical", external_curves)[[1]]
external_pirads = Filter(function(x) x[["model"]] == "pirads", external_curves)[[1]]
external_all_curves_except_pirads = Filter(function(x) !(x[["model"]] %in% c("pirads", "baseline_multi")), external_curves)
external_early_all_curves_except_pirads = Filter(function(x) x[["fusion"]] %in% c("early", "single"), external_all_curves_except_pirads)
external_late_all_curves_except_pirads = Filter(function(x) x[["fusion"]] %in% c("late", "single"), external_all_curves_except_pirads)
external_baseline_multi_lesion = Filter(function(x) x[["model"]] == "baseline_multi", external_curves)[[1]]

# Bootstrap sensitivity and specificity CIs for the PI-RADS scores
performance_pirads_internal = ci.thresholds(internal_pirads, boot.n=5000)
performance_pirads_external = ci.thresholds(external_pirads, boot.n=5000)

# Create a data frame containing the data for the PI-RADS error bar plots
pirads_vignette_internal = data.frame(
  sp.low = performance_pirads_internal$specificity[2:5,1],
  sp.median = performance_pirads_internal$specificity[2:5,2],
  sp.high = performance_pirads_internal$specificity[2:5,3],
  se.low = performance_pirads_internal$sensitivity[2:5,1],
  se.median = performance_pirads_internal$sensitivity[2:5,2],
  se.high = performance_pirads_internal$sensitivity[2:5,3],
  name = factor(c("PR\u22652", "PR\u22653", "PR\u22654", "PR\u22655"))
)
# Only keep PI-RADS 3 and 4 cutoffs
pirads_vignette_internal = subset(pirads_vignette_internal, name %in% c("PR\u22653", "PR\u22654"))

pirads_vignette_external = data.frame(
  sp.low = performance_pirads_external$specificity[2:5,1],
  sp.median = performance_pirads_external$specificity[2:5,2],
  sp.high = performance_pirads_external$specificity[2:5,3],
  se.low = performance_pirads_external$sensitivity[2:5,1],
  se.median = performance_pirads_external$sensitivity[2:5,2],
  se.high = performance_pirads_external$sensitivity[2:5,3],
  name = factor(c("PR\u22652", "PR\u22653", "PR\u22654", "PR\u22655"))
)
pirads_vignette_external = subset(pirads_vignette_external, name %in% c("PR\u22653", "PR\u22654"))


library(ggplot2)
library(ggsci)
library(ggnewscale)
library(ggrepel)
(
  internal_roc_plot = ggroc(internal_all_curves_except_pirads) +
    geom_abline(intercept = 1, slope = 1, linetype="dashed", color ="grey32", size=0.7) + 
    scale_color_hue(l = 40) +
    labs(color = "AI model") +
    new_scale_color() +
    scale_color_manual(values = c("#023361", "#026e41")) +
    geom_errorbar(aes(x=sp.median, ymax=se.high, ymin=se.low, color=name), data=pirads_vignette_internal, inherit.aes = F, size=0.8, width=0.02) +
    geom_errorbarh(aes(y=se.median, xmax=sp.high, xmin=sp.low, color=name), data=pirads_vignette_internal, inherit.aes = F, size=0.8, height=0.02) +
    geom_text(aes(y=se.median, x=sp.median, label=name), nudge_x = -0.09, nudge_y = 0.07, data=pirads_vignette_internal, inherit.aes = F) +
    theme_classic() +
    labs(color = "PI-RADS") +
    theme(
      panel.grid.major = element_line(color = "grey90",
                                      size = 0.5,
                                      linetype = 1),
      panel.grid.minor = element_line(color = "grey90",
                                      size = 0.5,
                                      linetype = 2),
      legend.position = "none"
    )
)
ggsave("~/internal_roc_curves.png", internal_roc_plot, width=4, height=4)

(
  external_roc_plot = ggroc(external_all_curves_except_pirads) +
  geom_abline(intercept = 1, slope = 1, linetype="dashed", color ="grey32", size=0.7) + 
    scale_color_hue(l = 40) +
  labs(color = "AI model") +
  new_scale_color() +
  scale_color_manual(values = c("#023361", "#026e41")) +
  geom_errorbar(aes(x=sp.median, ymax=se.high, ymin=se.low, color=name), data=pirads_vignette_external, inherit.aes = F, size=0.8, width=0.02) +
  geom_errorbarh(aes(y=se.median, xmax=sp.high, xmin=sp.low, color=name), data=pirads_vignette_external, inherit.aes = F, size=0.8, height=0.02) +
  geom_text(aes(y=se.median, x=sp.median, label=name), nudge_x = -0.1, nudge_y = 0.04, data=pirads_vignette_external, inherit.aes = F) +
  theme_classic() +
  labs(color = paste0("PI-RADS")) +
  theme(
    panel.grid.major = element_line(color = "grey90",
                                    size = 0.5,
                                    linetype = 1),
    panel.grid.minor = element_line(color = "grey90",
                                    size = 0.5,
                                    linetype = 2),
  )
)
ggsave("~/external_roc_curves.png", external_roc_plot, width=5.8, height=3.6)



(
  internal_early_roc_plot = ggroc(internal_early_all_curves_except_pirads) +
    geom_abline(intercept = 1, slope = 1, linetype="dashed", color ="grey32", size=0.7) + 
    scale_color_hue(l = 40) +
    labs(color = "AI model") +
    new_scale_color() +
    scale_color_manual(values = c("#023361", "#026e41")) +
    geom_errorbar(aes(x=sp.median, ymax=se.high, ymin=se.low, color=name), data=pirads_vignette_internal, inherit.aes = F, size=0.8, width=0.02) +
    geom_errorbarh(aes(y=se.median, xmax=sp.high, xmin=sp.low, color=name), data=pirads_vignette_internal, inherit.aes = F, size=0.8, height=0.02) +
    geom_text(aes(y=se.median, x=sp.median, label=name), nudge_x = -0.09, nudge_y = 0.07, data=pirads_vignette_internal, inherit.aes = F) +
    theme_classic() +
    labs(color = "PI-RADS") +
    theme(
      panel.grid.major = element_line(color = "grey90",
                                      size = 0.5,
                                      linetype = 1),
      panel.grid.minor = element_line(color = "grey90",
                                      size = 0.5,
                                      linetype = 2),
      legend.position = "none"
    )
)
ggsave("~/internal_early_roc_curves.png", internal_early_roc_plot, width=4, height=4)

(
  internal_late_roc_plot = ggroc(internal_late_all_curves_except_pirads) +
    geom_abline(intercept = 1, slope = 1, linetype="dashed", color ="grey32", size=0.7) + 
    scale_color_hue(l = 40) +
    labs(color = "AI model") +
    new_scale_color() +
    scale_color_manual(values = c("#023361", "#026e41")) +
    geom_errorbar(aes(x=sp.median, ymax=se.high, ymin=se.low, color=name), data=pirads_vignette_internal, inherit.aes = F, size=0.8, width=0.02) +
    geom_errorbarh(aes(y=se.median, xmax=sp.high, xmin=sp.low, color=name), data=pirads_vignette_internal, inherit.aes = F, size=0.8, height=0.02) +
    geom_text(aes(y=se.median, x=sp.median, label=name), nudge_x = -0.09, nudge_y = 0.07, data=pirads_vignette_internal, inherit.aes = F) +
    theme_classic() +
    labs(color = "PI-RADS") +
    theme(
      panel.grid.major = element_line(color = "grey90",
                                      size = 0.5,
                                      linetype = 1),
      panel.grid.minor = element_line(color = "grey90",
                                      size = 0.5,
                                      linetype = 2),
      legend.position = "none"
    )
)
ggsave("~/internal_late_roc_curves.png", internal_late_roc_plot, width=4, height=4)

(
  external_early_roc_plot = ggroc(external_early_all_curves_except_pirads) +
    geom_abline(intercept = 1, slope = 1, linetype="dashed", color ="grey32", size=0.7) + 
    scale_color_hue(l = 40) +
    labs(color = "AI model") +
    new_scale_color() +
    scale_color_manual(values = c("#023361", "#026e41")) +
    geom_errorbar(aes(x=sp.median, ymax=se.high, ymin=se.low, color=name), data=pirads_vignette_external, inherit.aes = F, size=0.8, width=0.02) +
    geom_errorbarh(aes(y=se.median, xmax=sp.high, xmin=sp.low, color=name), data=pirads_vignette_external, inherit.aes = F, size=0.8, height=0.02) +
    geom_text(aes(y=se.median, x=sp.median, label=name), nudge_x = -0.09, nudge_y = 0.07, data=pirads_vignette_external, inherit.aes = F) +
    theme_classic() +
    labs(color = "PI-RADS") +
    theme(
      panel.grid.major = element_line(color = "grey90",
                                      size = 0.5,
                                      linetype = 1),
      panel.grid.minor = element_line(color = "grey90",
                                      size = 0.5,
                                      linetype = 2),
      legend.position = "none"
    )
)
ggsave("~/external_early_roc_curves.png", external_early_roc_plot, width=4, height=4)

(
  external_late_roc_plot = ggroc(external_late_all_curves_except_pirads) +
    geom_abline(intercept = 1, slope = 1, linetype="dashed", color ="grey32", size=0.7) + 
    scale_color_hue(l = 40) +
    labs(color = "AI model") +
    new_scale_color() +
    scale_color_manual(values = c("#023361", "#026e41")) +
    geom_errorbar(aes(x=sp.median, ymax=se.high, ymin=se.low, color=name), data=pirads_vignette_external, inherit.aes = F, size=0.8, width=0.02) +
    geom_errorbarh(aes(y=se.median, xmax=sp.high, xmin=sp.low, color=name), data=pirads_vignette_external, inherit.aes = F, size=0.8, height=0.02) +
    geom_text(aes(y=se.median, x=sp.median, label=name), nudge_x = -0.09, nudge_y = 0.07, data=pirads_vignette_external, inherit.aes = F) +
    theme_classic() +
    labs(color = "PI-RADS") +
    theme(
      panel.grid.major = element_line(color = "grey90",
                                      size = 0.5,
                                      linetype = 1),
      panel.grid.minor = element_line(color = "grey90",
                                      size = 0.5,
                                      linetype = 2),
      legend.position = "none"
    )
)
ggsave("~/external_late_roc_curves.png", external_late_roc_plot, width=4, height=4)

# Experiment 1. Determine optimal model internally
max_auc_index <- which.max(sapply(internal_ai_curves, function(x) x$auc))
(max_auc_name <- names(internal_ai_curves)[max_auc_index])
best_internal_curve = internal_ai_curves[[max_auc_name]]
ci.auc(best_internal_curve)

# Compare best model performance to baseline models
(roctest_1 = roc.test(best_internal_curve, internal_baseline))
(adjusted_p1 = p.adjust(roctest_1$p.value, method="holm", n=10))

(roctest_2 = roc.test(best_internal_curve, internal_clinical))
(adjusted_p2 = p.adjust(roctest_2$p.value, method="holm", n=10))

(roctest_3 = roc.test(best_internal_curve, internal_pirads))
(adjusted_p3 = p.adjust(roctest_3$p.value, method="holm", n=10))

# Was there a significant difference with early fusion?
internal_alternative_fusion_curve = find_curve(internal_curves, "internal", best_internal_curve$model, other_fusion_name(best_internal_curve$fusion))
(roctest_4 = roc.test(best_internal_curve, internal_alternative_fusion_curve))
(adjusted_p4 = p.adjust(roctest_4$p.value, method="holm", n=10))

# Was there a significant difference with inclusion of volumetric lesion information?
internal_with_volumetric_fusion_curve = find_curve(internal_curves, "internal", "ai_with_clinical", best_internal_curve$fusion)
(roctest_5 = roc.test(best_internal_curve, internal_with_volumetric_fusion_curve))
(adjusted_p5 = p.adjust(roctest_5$p.value, method="holm", n=10))


# Experiment 2. External validation
# Apply best internal model to external dataset
best_external_curve = find_curve(external_curves, "external", best_internal_curve$model, best_internal_curve$fusion)
ci.auc(best_external_curve)

# Compare best model performance to baseline models
(roctest_6 = roc.test(best_external_curve, external_clinical))
(adjusted_p6 = p.adjust(roctest_6$p.value, method="holm", n=10))

(roctest_7 = roc.test(best_external_curve, external_baseline))
(adjusted_p7 = p.adjust(roctest_7$p.value, method="holm", n=10))

# Was early fusion significantly better than late fusion?
external_alternative_fusion_curve = find_curve(external_curves, "external", best_internal_curve$model, other_fusion_name(best_internal_curve$fusion))
(roctest_8 = roc.test(best_external_curve, external_alternative_fusion_curve))
(adjusted_p8 = p.adjust(roctest_8$p.value, method="holm", n=10))

# Was there a significant difference with inclusion of volumetric lesion information?
external_with_volumetric_fusion_curve = find_curve(external_curves, "external", "ai_with_clinical", best_internal_curve$fusion)
(roctest_9 = roc.test(best_external_curve, external_with_volumetric_fusion_curve))
(adjusted_p9 = p.adjust(roctest_9$p.value, method="holm", n=10))

# Radiologist performance 
internal_pirads$auc 
ci.auc(internal_pirads)
external_pirads$auc
ci.auc(external_pirads)

(roctest_10 = roc.test(best_external_curve, external_pirads))
(adjusted_p10 = p.adjust(roctest_10$p.value, method="holm", n=10))
(roctest_11 = roc.test(best_internal_curve, internal_pirads))
(adjusted_p11 = p.adjust(roctest_11$p.value, method="holm", n=10))

pirads_internal_dataset = subset(data, model=="pirads" & dataset=="internal")
pirads_external_dataset = subset(data, model=="pirads" & dataset=="external")

#####################################
####      JACKKNIFE ANALYSIS     ####
#####################################

# Only visualization, actual jackknife AUCs are obtained using jackknife.py

jk_results = data.frame()

# Internal Without
jk_results = rbind(jk_results, data.frame(parameter="primary lesion suspicion", type="without variable", dataset="internal", AUC=0.7213114754098361))
jk_results = rbind(jk_results, data.frame(parameter="secondary lesion suspicion", type="without variable", dataset="internal", AUC=0.8729508196721311))
jk_results = rbind(jk_results, data.frame(parameter="tertiary lesion suspicion", type="without variable", dataset="internal", AUC=0.8737704918032787))
jk_results = rbind(jk_results, data.frame(parameter="PSA level", type="without variable", dataset="internal", AUC=0.8254098360655738))
jk_results = rbind(jk_results, data.frame(parameter="prostate volume", type="without variable", dataset="internal", AUC=0.8516393442622952))
jk_results = rbind(jk_results, data.frame(parameter="patient age", type="without variable", dataset="internal", AUC=0.8688524590163934))

# Internal with only
jk_results = rbind(jk_results, data.frame(parameter="primary lesion suspicion", type="with only variable", dataset="internal", AUC=0.7860655737704918))
jk_results = rbind(jk_results, data.frame(parameter="secondary lesion suspicion", type="with only variable", dataset="internal", AUC=0.5413934426229507))
jk_results = rbind(jk_results, data.frame(parameter="tertiary lesion suspicion", type="with only variable", dataset="internal", AUC=0.525))
jk_results = rbind(jk_results, data.frame(parameter="PSA level", type="with only variable", dataset="internal", AUC=0.6565573770491804))
jk_results = rbind(jk_results, data.frame(parameter="prostate volume", type="with only variable", dataset="internal", AUC=0.5168032786885246))
jk_results = rbind(jk_results, data.frame(parameter="patient age", type="with only variable", dataset="internal", AUC=0.6807377049180328))

# Internal all variables
jk_results = rbind(jk_results, data.frame(parameter="all variables", type="with all variables", dataset="internal", AUC=0.8737704918032787))



# External Without
jk_results = rbind(jk_results, data.frame(parameter="primary lesion suspicion", type="without variable", dataset="external", AUC=0.6866661897778892))
jk_results = rbind(jk_results, data.frame(parameter="secondary lesion suspicion", type="without variable", dataset="external", AUC=0.7762259022139563))
jk_results = rbind(jk_results, data.frame(parameter="tertiary lesion suspicion", type="without variable", dataset="external", AUC=0.7748131192102722))
jk_results = rbind(jk_results, data.frame(parameter="PSA level", type="without variable", dataset="external", AUC=0.7579491398118675))
jk_results = rbind(jk_results, data.frame(parameter="prostate volume", type="without variable", dataset="external", AUC=0.7640652383847777))
jk_results = rbind(jk_results, data.frame(parameter="patient age", type="without variable", dataset="external", AUC=0.7753764440788297))

# External with only
jk_results = rbind(jk_results, data.frame(parameter="primary lesion suspicion", type="with only variable", dataset="external", AUC=0.7502145999499267))
jk_results = rbind(jk_results, data.frame(parameter="secondary lesion suspicion", type="with only variable", dataset="external", AUC=0.5687792839515003))
jk_results = rbind(jk_results, data.frame(parameter="tertiary lesion suspicion", type="with only variable", dataset="external", AUC=0.49629815086376483))
jk_results = rbind(jk_results, data.frame(parameter="PSA level", type="with only variable", dataset="external", AUC=0.5926535283808434))
jk_results = rbind(jk_results, data.frame(parameter="prostate volume", type="with only variable", dataset="external", AUC=0.6232966128974569))
jk_results = rbind(jk_results, data.frame(parameter="patient age", type="with only variable", dataset="external", AUC=0.5707017418362602))

# External all variables
jk_results = rbind(jk_results, data.frame(parameter="all variables", type="with all variables", dataset="external", AUC=0.7742229693479739))


jk_results$parameter.fac = factor(jk_results$parameter, levels = c(
  "all variables",
  "tertiary lesion suspicion",
  "secondary lesion suspicion",
  "patient age",
  "prostate volume",
  "PSA level",
  "primary lesion suspicion"
))
jk_results$type.fac = factor(jk_results$type, levels=c("without variable", "with only variable", "with all variables"))
jk_results$dataset.fac = factor(jk_results$dataset)


(internal_jackknife_plot = ggplot(subset(jk_results, dataset=="internal"), aes(y=parameter.fac, x=AUC, fill=type.fac)) + 
    geom_bar(
      stat="identity", 
      position=position_nudge(y=ifelse(jk_results$type.fac == "with only variable", 0.1, 0)),
      width=0.5) +
    coord_cartesian(xlim=c(0.5, 0.9))+ 
    scale_x_continuous(breaks=c(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)) +
    scale_fill_manual(values=c("#00AAA9", "#0001FC", "red")) +
    labs(y="") +
    theme_classic() +
    theme(
      legend.title=element_blank(),
      panel.grid.major = element_line(color = "grey80",
                                      size = 0.3,
                                      linetype = 1),
      panel.border = element_rect(color = "black", 
                                  fill = NA, 
                                  size = 0.5)
    ) 
)
ggsave("~/internal_jackknife.png", internal_jackknife_plot, width=6.5, height=2.6, dpi=600)

(external_jackknife_plot = ggplot(subset(jk_results, dataset=="external"), aes(y=parameter.fac, x=AUC, fill=type.fac)) + 
    geom_bar(
      stat="identity", 
      position=position_nudge(y=ifelse(jk_results$type.fac == "with only variable", 0.1, 0)),
      width=0.5) +
    coord_cartesian(xlim=c(0.5, 0.8)) +
    scale_x_continuous(breaks=c(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8), expand = c(0.05, 0.00)) +
    scale_fill_manual(values=c("#00AAA9", "#0001FC", "red")) +
    labs(y="") +
    theme_classic() +
    theme(
      legend.title=element_blank(),
      panel.grid.major = element_line(color = "grey80",
                                      size = 0.3,
                                      linetype = 1),
      panel.border = element_rect(color = "black", 
                                  fill = NA, 
                                  size = 0.5)
    ) 
)

ggsave("~/external_jackknife.png", external_jackknife_plot, width=6.5, height=2.6, dpi=600)
