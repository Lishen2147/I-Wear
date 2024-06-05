"use client";

import React from "react";
import ProbabilityItem from "./subcomponents/Upload/ProbabilityItem";

interface PredictionProbabiltiesProps {
  probabilities: number[] | null;
}

const PredictionProbabilties: React.FC<PredictionProbabiltiesProps> = ({
  probabilities,
}) => {
  if (!probabilities) {
    return <></>;
  } else {
    return (
      <div className="flex flex-col items-center gap-8 p-12">
        <div className="flex flex-col items-center gap-2">
          <h1 className="text-3xl font-semibold text-black">
            Information Metrics
          </h1>
          <p className="text-lg">
            Obtained through our high performing custom model
            <span className="font-bold text-lg">*</span>
          </p>
        </div>
        <div className="flex flex-wrap gap-4">
          {probabilities?.map((prob, idx) => (
            <ProbabilityItem key={idx} probability={prob} />
          ))}
        </div>
        <div>
          <p>
            * - we may have overexaggerated the facts. it was a pre-structured
            model that we tuned ourselves, but the results are still pretty
            good.
          </p>
        </div>
      </div>
    );
  }
};

export default PredictionProbabilties;
