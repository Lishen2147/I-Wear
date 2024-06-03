"use client";

import React from "react";
import ProbabilityItem from "./subcomponents/Upload/ProbabilityItem";

interface PredictionProbabiltiesProps {
    classes: string[],
    probabilities: number[] | null
}


const PredictionProbabilties: React.FC<PredictionProbabiltiesProps> = ({classes, probabilities}) => {
  return (
    <div>
       {probabilities?.map((prob, idx) => (
  <ProbabilityItem key={idx} predictedClassName={classes[idx]} probability={prob} classNumber={idx+1} />
))}
    </div>
  );
};

export default PredictionProbabilties;
