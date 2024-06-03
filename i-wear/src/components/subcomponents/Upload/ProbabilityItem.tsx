"use client";

import React from "react";

interface ProbabilityItemProps {
    predictedClassName: string,
    probability: number,
    classNumber: number
}

const ProbabilityItem: React.FC<ProbabilityItemProps> = ({predictedClassName, probability, classNumber}) => {
  return (
    <div>
        <div>{"Class " + classNumber + ": " + predictedClassName}</div>
        <div>{"Probability: " + probability}</div>
    </div>
  );
};

export default ProbabilityItem;
