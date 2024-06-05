"use client";

import React from "react";

interface ProbabilityItemProps {
  probability: number;
}

const ProbabilityItem: React.FC<ProbabilityItemProps> = ({ probability }) => {
  const infoData = [
    {
      title: "Facial Class",
      value: probability[0],
    },
    {
      title: "Probability",
      value: `${(probability[1] * 100).toFixed(2)}%`,
    },
  ];

  return (
    <div className="flex flex-col justify-evenly p-4 gap-2 rounded-lg bg-slate-600 min-h-[150px] min-w-[300px] transition-all duration-300 hover:bg-black">
      {infoData.map((data) => (
        <div key={data.title} className="flex flex-col items-center gap-1">
          <div className="text-xl text-white underline underline-offset-4 decoration-orange-400 decoration-3">
            {data.title}
          </div>
          <div className="text-lg text-white">{data.value}</div>
        </div>
      ))}
    </div>
  );
};

export default ProbabilityItem;
